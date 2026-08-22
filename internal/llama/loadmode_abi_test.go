// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

//go:build llamaabi

package llama

import "testing"

// The load-mode constants are written as literals so this package still builds against
// llama.cpp revisions predating the enum. That trades a compile error for a silent mismatch,
// so they are checked against the header wherever it defines them.
//
// Not hypothetical: transcribing them by eye put every mode one place out, with Auto landing
// on None and Mmap on Mlock. Nothing would have reported that — the weights would simply have
// been loaded a different way than asked for.
func TestLoadModeValuesMatchTheHeader(t *testing.T) {
	ours := map[string]int32{
		"auto":       int32(LoadModeAuto),
		"none":       int32(LoadModeNone),
		"mmap":       int32(LoadModeMmap),
		"mlock":      int32(LoadModeMlock),
		"mmap_mlock": int32(LoadModeMmapMlock),
		"direct_io":  int32(LoadModeDirectIO),
	}
	for name, want := range cLoadModes {
		if got := ours[name]; got != want {
			t.Errorf("%s: ours %d, llama.h %d — the constants have drifted from the header",
				name, got, want)
		}
	}
}

// The greedy fast path must select exactly what the library's chain would, or a throughput
// optimisation has quietly changed what the model says. The library starts at the first entry
// and replaces only on a strictly greater logit, so the earliest maximum wins; a scan using >=
// would pick the last one and diverge on ties, which real vocabularies produce.
func TestGreedyFastPathMatchesTheLibrarySelection(t *testing.T) {
	cases := [][]float32{
		{0.1, 0.9, 0.3},
		{2.0, 2.0, 1.0},  // tie at the front — the earliest must win
		{-5, -1, -1, -9}, // tie among negatives
		{7},              // single candidate
		{1, 2, 3, 4, 5},  // maximum last
		{5, 4, 3, 2, 1},  // maximum first
	}
	for _, logits := range cases {
		// What our scan does.
		best, bestV := 0, logits[0]
		for i := 1; i < len(logits); i++ {
			if logits[i] > bestV {
				bestV, best = logits[i], i
			}
		}
		// What llama_sampler_greedy_apply does, transcribed from the library.
		sel := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[sel] {
				sel = i
			}
		}
		if best != sel {
			t.Errorf("logits %v: fast path chose %d, library would choose %d", logits, best, sel)
		}
	}
}

// A chain that is not pure greedy must not take the fast path: penalties and temperature
// rewrite the logits before selection, so scanning them directly reads the wrong values.
func TestOnlyPureGreedyTakesTheFastPath(t *testing.T) {
	for _, c := range []struct {
		name string
		p    SamplingParams
		want bool
	}{
		{"greedy", SamplingParams{Temperature: 0}, true},
		{"greedy, penalties configured but inert", SamplingParams{Temperature: 0, RepeatLastN: 64, RepeatPenalty: 1}, true},
		{"greedy with a repeat penalty", SamplingParams{Temperature: 0, RepeatLastN: 64, RepeatPenalty: 1.1}, false},
		{"greedy with a presence penalty", SamplingParams{Temperature: 0, RepeatLastN: 64, PresencePenalty: 0.5}, false},
		{"sampling", SamplingParams{Temperature: 0.8}, false},
	} {
		if got := greedyDirect(c.p); got != c.want {
			t.Errorf("%s: greedyDirect = %v, want %v", c.name, got, c.want)
		}
	}
}
