// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

// standby answers health checks while the deployment is still getting ready.
//
// Two problems, one cause. A container that downloads a 14 GiB model, benchmarks the library and
// sweeps configurations before it can serve takes the better part of an hour, during which
// nothing is listening: the platform's health check fails, the instance is declared unhealthy
// and reallocated, and the boot starts over — which was observed twice in one night, each time
// discarding around fifty minutes of download. And because these hosts offer no way to read a
// container's stdout, there is no way to tell a slow boot from a hung one.
//
// So bind the port immediately and say what is happening. The health check passes because
// something is answering, and /v1/info reports the phase rather than nothing at all. The real
// server replaces this process once the work is done.
func standby(args []string) int {
	fs := newFlagSet("standby")
	addr := fs.String("addr", ":8080", "address to hold")
	statusPath := fs.String("status", "", "file whose contents describe the current phase")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "standby: %v\n", err)
		return 1
	}
	started := time.Now()

	phase := func() string {
		if *statusPath == "" {
			return "starting"
		}
		b, err := os.ReadFile(*statusPath)
		if err != nil {
			return "starting"
		}
		if s := strings.TrimSpace(string(b)); s != "" {
			return s
		}
		return "starting"
	}

	mux := http.NewServeMux()
	// Health must succeed, or the point is lost: the platform replaces the instance and the
	// download starts again. This says "alive and working", which is true.
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeStatus(w, http.StatusOK, phase(), started)
	})
	// Everything else is honestly unavailable — a caller must not mistake standby for a
	// server that can answer, and 503 is what a client should retry against.
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeStatus(w, http.StatusServiceUnavailable, phase(), started)
	})

	srv := &http.Server{Handler: mux}
	go func() { _ = srv.Serve(ln) }()
	fmt.Printf("standby: holding %s while the deployment prepares\n", *addr)

	// Exit on the signal the entrypoint sends when the real server is ready to take the port.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
	<-sig
	_ = srv.Close()
	_ = ln.Close()
	return 0
}

func writeStatus(w http.ResponseWriter, code int, phase string, started time.Time) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":          "starting",
		"phase":           phase,
		"elapsed_seconds": int(time.Since(started).Seconds()),
		"detail": "this deployment is preparing — downloading the model, measuring the " +
			"library, or sweeping configurations. It is not yet able to serve requests.",
	})
}
