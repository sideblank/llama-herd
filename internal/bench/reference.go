// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package bench

import "fmt"

// Reference is a configuration that was actually measured on real hardware, kept in code rather
// than only in documentation.
//
// The point is that a planner, a `fit` calculation or a default can consult what was measured
// instead of what seems reasonable. Every field here came off a card; nothing is projected. When
// a figure is re-measured, change it here and the places that read it follow.
type Reference struct {
	// Card and Model identify what was measured. A reference is worthless without both.
	Card  string
	Model string
	// Quant is the exact build, because throughput moves with it.
	Quant string
	// LlamaCppRef and ForceMMQ pin the inference build. FORCE_MMQ measured 7-9% on this card
	// and is not the upstream default, so a reader reproducing these numbers needs it.
	LlamaCppRef string
	ForceMMQ    bool

	// Streams is the configuration to ship: the best measured point that is not adjacent to a
	// capacity cliff. It is not always the fastest point — see PeakStreams.
	Streams uint32
	// TotalContext is the KV pool shared across those streams; each gets TotalContext/Streams.
	TotalContext uint32
	KVTypeK      string
	KVTypeV      string
	FlashAttn    bool
	Batch        uint32

	// AggregateTokPerSec is what Streams delivered, decoding with a nearly empty cache.
	AggregateTokPerSec float64
	// LibraryTokPerSec is llama-bench's tg128 on the SAME boot. Every throughput figure here is
	// only comparable through this: the same configuration measured 564 and 728 on two rented
	// 3090s whose library figures were 119.31 and 138.40.
	LibraryTokPerSec float64

	// PeakStreams is the fastest point measured, and PeakTokPerSec what it gave. Shipping it is
	// usually wrong: on the 3090 it beat the shipped setting by 3% while sitting one step from
	// a configuration that collapsed.
	PeakStreams   uint32
	PeakTokPerSec float64
	// CliffStreams is the lowest stream count observed to fail or collapse. Its position moves
	// with the node, so treat it as a warning boundary rather than a limit to design against.
	CliffStreams uint32

	// DepthNote records that these are shallow-cache figures, since that is the caveat most
	// likely to be dropped when a number is quoted.
	DepthNote string
}

// Amortisation is how much more work the herd retires than the library manages at one sequence,
// on the same boot. This is the number that survives moving between machines; the raw throughput
// is not.
func (r Reference) Amortisation() float64 {
	if r.LibraryTokPerSec == 0 {
		return 0
	}
	return r.AggregateTokPerSec / r.LibraryTokPerSec
}

// ContextPerStream is what one sequence gets out of the shared pool.
func (r Reference) ContextPerStream() uint32 {
	if r.Streams == 0 {
		return 0
	}
	return r.TotalContext / r.Streams
}

// Summary is a one-paragraph account suitable for printing beside a plan.
func (r Reference) Summary() string {
	return fmt.Sprintf(
		"measured on %s with %s (%s): %d streams x %d context = %.0f tok/s aggregate, "+
			"%.2fx the library's %.0f tok/s on the same boot. Peak was %d streams (%.0f tok/s, "+
			"+%.0f%%) but %d streams collapsed. %s",
		r.Card, r.Model, r.Quant, r.Streams, r.ContextPerStream(), r.AggregateTokPerSec,
		r.Amortisation(), r.LibraryTokPerSec, r.PeakStreams, r.PeakTokPerSec,
		(r.PeakTokPerSec/r.AggregateTokPerSec-1)*100, r.CliffStreams, r.DepthNote)
}

// References are the configurations measured to date, keyed by card.
//
// Only the 3090 has been measured. The 4090 and 5090 entries in the roadmap are projections and
// are deliberately absent here — a table of measurements must not carry estimates, or the next
// reader cannot tell which is which.
var References = map[string]Reference{
	"3090": {
		Card:        "3090",
		Model:       "Qwen3.6-35B-A3B",
		Quant:       "UD-IQ3_S",
		LlamaCppRef: "b10545",
		ForceMMQ:    true,

		Streams:      48,
		TotalContext: 425984,
		KVTypeK:      "q8_0",
		KVTypeV:      "q8_0",
		FlashAttn:    true,
		Batch:        2048,

		// Measured 2026-08-23 on one node with 48 as the control point in the same sweep.
		// A different node the same day gave 728.71 against a library figure of 138.40 —
		// 5.27x, against 4.73x here. Same behaviour, faster hardware.
		AggregateTokPerSec: 564.16,
		LibraryTokPerSec:   119.31,

		PeakStreams:   64,
		PeakTokPerSec: 582.04,
		CliffStreams:  72,

		DepthNote: "shallow cache; the same card sustains roughly 110 tok/s at 16k per stream, " +
			"the deepest herd that fits is 24 streams and 8 beats it by more than 2x",
	},
}

// ReferenceFor returns the measured configuration for a card, if one exists.
func ReferenceFor(card string) (Reference, bool) {
	r, ok := References[card]
	return r, ok
}
