// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Command llama-herd is the runtime's entry point.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	"github.com/sideblank/llama-herd/internal/llama"
)

// Set at link time by the release build:
//
//	-ldflags "-X main.version=v1.2.3 -X main.commit=abc1234 -X main.date=..."
var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
)

func usage() {
	fmt.Fprintf(os.Stderr, `llama-herd %s

usage:
  llama-herd version     print version and build information
  llama-herd doctor      verify the llama.cpp library loads and links

`, version)
}

func main() {
	flag.Usage = usage
	flag.Parse()

	switch flag.Arg(0) {
	case "version":
		fmt.Printf("llama-herd %s\n", version)
		fmt.Printf("  commit:   %s\n", commit)
		fmt.Printf("  built:    %s\n", date)
		fmt.Printf("  go:       %s\n", runtime.Version())
		fmt.Printf("  platform: %s/%s\n", runtime.GOOS, runtime.GOARCH)

	case "doctor":
		// Proves the binary actually links and initialises libllama, which is the
		// failure a release archive is most likely to ship broken: a binary that
		// builds fine but cannot find its library at run time.
		llama.Backend()
		defer llama.BackendFree()
		fmt.Printf("llama.cpp backend initialised — linkage OK (%s/%s)\n", runtime.GOOS, runtime.GOARCH)

	default:
		usage()
		os.Exit(2)
	}
}
