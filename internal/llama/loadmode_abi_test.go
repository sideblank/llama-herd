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
