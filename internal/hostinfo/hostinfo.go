// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package hostinfo reports what the machine is doing.
//
// Everything here is best-effort and degrades to zero rather than failing. The binary ships
// for Linux, macOS and Windows, and a monitoring endpoint that errors on a platform where one
// counter is unavailable is worse than one that omits it.
package hostinfo

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
)

var start = time.Now()

// Host is a snapshot of the machine.
type Host struct {
	OS   string `json:"os"`
	Arch string `json:"arch"`
	CPUs int    `json:"cpus"`

	// LoadAvg1 is the one-minute load average, or 0 where unavailable. Read as a fraction
	// of CPUs: sustained load above the core count means the machine is oversubscribed and
	// inference is competing for CPU it will not get.
	LoadAvg1  float64 `json:"load_avg_1m"`
	LoadAvg5  float64 `json:"load_avg_5m"`
	LoadAvg15 float64 `json:"load_avg_15m"`

	// Memory in bytes, 0 where unavailable.
	MemTotalBytes     uint64 `json:"mem_total_bytes"`
	MemAvailableBytes uint64 `json:"mem_available_bytes"`

	// Process figures, always available.
	ProcHeapBytes uint64 `json:"proc_heap_bytes"`
	ProcSysBytes  uint64 `json:"proc_sys_bytes"`
	Goroutines    int    `json:"goroutines"`

	UptimeSeconds float64 `json:"uptime_seconds"`
}

// Read returns the current snapshot.
func Read() Host {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	h := Host{
		OS:            runtime.GOOS,
		Arch:          runtime.GOARCH,
		CPUs:          runtime.NumCPU(),
		ProcHeapBytes: ms.HeapAlloc,
		ProcSysBytes:  ms.Sys,
		Goroutines:    runtime.NumGoroutine(),
		UptimeSeconds: time.Since(start).Seconds(),
	}
	h.LoadAvg1, h.LoadAvg5, h.LoadAvg15 = loadAvg()
	h.MemTotalBytes, h.MemAvailableBytes = memory()
	return h
}

// Oversubscribed reports whether the machine is loaded beyond its cores, the state that makes
// throughput collapse for reasons unrelated to the model.
func (h Host) Oversubscribed() bool {
	return h.CPUs > 0 && h.LoadAvg1 > float64(h.CPUs)
}

func loadAvg() (float64, float64, float64) {
	b, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return 0, 0, 0 // not Linux, or unreadable
	}
	f := strings.Fields(string(b))
	if len(f) < 3 {
		return 0, 0, 0
	}
	p := func(s string) float64 { v, _ := strconv.ParseFloat(s, 64); return v }
	return p(f[0]), p(f[1]), p(f[2])
}

func memory() (total, available uint64) {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		f := strings.Fields(line)
		if len(f) < 2 {
			continue
		}
		v, err := strconv.ParseUint(f[1], 10, 64) // values are in kB
		if err != nil {
			continue
		}
		switch f[0] {
		case "MemTotal:":
			total = v * 1024
		case "MemAvailable:":
			available = v * 1024
		}
	}
	return total, available
}
