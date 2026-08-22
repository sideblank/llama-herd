// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

import (
	"runtime"
	"testing"
)

func TestAutoThreadsSuitsWhereTheWorkIs(t *testing.T) {
	cpus := runtime.NumCPU()

	gpu := AutoThreads(true)
	if gpu < 1 || int(gpu) > cpus {
		t.Fatalf("offloaded: %d threads on a %d-core host is not sensible", gpu, cpus)
	}
	if cpus > 4 && gpu != 4 {
		t.Errorf("offloaded on %d cores should stay modest, got %d", cpus, gpu)
	}

	cpu := AutoThreads(false)
	if cpu < 1 || int(cpu) > cpus {
		t.Fatalf("on CPU: %d threads on a %d-core host is not sensible", cpu, cpus)
	}
	if cpus > 1 && int(cpu) != cpus-1 {
		t.Errorf("on CPU should use %d of %d cores, leaving one for the system, got %d",
			cpus-1, cpus, cpu)
	}
	if cpus > 4 && cpu <= gpu {
		t.Errorf("CPU inference should ask for more threads than an offloaded model: %d vs %d",
			cpu, gpu)
	}
}

// The logit buffer is sized by what will be requested, not by the prefill chunk. Getting this
// wrong costs device memory measured in gigabytes on a large vocabulary.
func TestAutoOutputsMaxBoundsTheLogitBuffer(t *testing.T) {
	for _, c := range []struct {
		name             string
		streams, draft   int
		outputEverywhere bool
		want             uint32
	}{
		{"four streams, no drafting", 4, 0, false, 8}, // 4 needed, floor applies
		{"four streams, two drafts", 4, 2, false, 12}, // 4 x (1+2)
		{"six streams, four drafts", 6, 4, false, 30}, // 6 x (1+4)
		{"one stream", 1, 0, false, 8},                // floor
		{"hidden states needed", 4, 2, true, 0},       // must hold a whole chunk
	} {
		if got := AutoOutputsMax(c.streams, c.draft, c.outputEverywhere); got != c.want {
			t.Errorf("%s: got %d, want %d", c.name, got, c.want)
		}
	}

	// The saving is the point: a 2048 batch against a 150k vocabulary reserves over a
	// gigabyte, where twelve outputs need a few megabytes.
	const vocab, batch = 151936, 2048
	full := uint64(batch) * vocab * 4
	tuned := uint64(AutoOutputsMax(4, 2, false)) * vocab * 4
	if tuned*50 > full {
		t.Fatalf("tuned buffer %d is not meaningfully smaller than the default %d", tuned, full)
	}
	t.Logf("logit buffer: %.2f GiB default -> %.1f MiB tuned",
		float64(full)/(1<<30), float64(tuned)/(1<<20))
}
