// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package bench

import "testing"

// The reference table is the only place a measured number lives in code, so the things that make
// it trustworthy are asserted rather than assumed.
func TestReferenceTableIsInternallyConsistent(t *testing.T) {
	for card, r := range References {
		if r.LibraryTokPerSec <= 0 {
			t.Errorf("%s: no library figure — a throughput number without the library's own "+
				"reading from the same boot cannot be compared across machines", card)
		}
		if r.AggregateTokPerSec <= r.LibraryTokPerSec {
			t.Errorf("%s: aggregate %.2f is not above the library's single-sequence %.2f, "+
				"which would mean the herd bought nothing", card, r.AggregateTokPerSec,
				r.LibraryTokPerSec)
		}
		if r.Streams == 0 || r.TotalContext == 0 {
			t.Errorf("%s: incomplete configuration", card)
			continue
		}
		// The shipped setting must leave room to generate, and must sit below the cliff.
		if r.CliffStreams > 0 && r.Streams >= r.CliffStreams {
			t.Errorf("%s: ships %d streams at or past the observed cliff of %d",
				card, r.Streams, r.CliffStreams)
		}
		// Shipping the peak is allowed, but only when it is not adjacent to collapse. Here it
		// is, which is why the two fields are separate.
		if r.PeakStreams > 0 && r.PeakTokPerSec < r.AggregateTokPerSec {
			t.Errorf("%s: peak %.2f is below the shipped figure %.2f",
				card, r.PeakTokPerSec, r.AggregateTokPerSec)
		}
		if r.DepthNote == "" {
			t.Errorf("%s: no depth note — a shallow-cache figure quoted without that caveat "+
				"overstates what the configuration does on real traffic", card)
		}
	}
}

// The 3090 numbers are what the campaign produced. Pinning them means a change has to be
// deliberate, and shows up in review as a claim about hardware rather than a silent edit.
func TestReference3090MatchesWhatWasMeasured(t *testing.T) {
	r, ok := ReferenceFor("3090")
	if !ok {
		t.Fatal("no 3090 reference")
	}
	if r.Streams != 48 || r.TotalContext != 425984 {
		t.Errorf("shipped configuration changed: %d streams x %d", r.Streams, r.TotalContext)
	}
	if got := r.ContextPerStream(); got != 8874 {
		t.Errorf("context per stream = %d, want 8874", got)
	}
	if got := r.Amortisation(); got < 4.5 || got > 5.5 {
		t.Errorf("amortisation = %.2f, want 4.5-5.5 (measured 4.73 and 5.27 on two nodes)", got)
	}
	// 64 streams beat 48 by only 3%, which is the reason 48 ships.
	gain := (r.PeakTokPerSec/r.AggregateTokPerSec - 1) * 100
	if gain > 10 {
		t.Errorf("peak now beats the shipped setting by %.1f%% — if that is real, ship the "+
			"peak instead", gain)
	}
}
