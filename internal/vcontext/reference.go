// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import "github.com/sideblank/llama-herd/internal/bench"

// PolicyForCard builds a policy from what was actually measured on a card.
//
// This is the join between the measurement table and the planner: the stream count and context
// share come from a real deployment rather than from a guess, so a change to the measured
// configuration moves the planner with it instead of leaving the two to drift.
//
// Reports false for a card with no measurements. A planner configured from projections would
// produce plans nothing has ever run.
func PolicyForCard(card string, shape Shape) (Policy, bool) {
	ref, ok := bench.ReferenceFor(card)
	if !ok {
		return Policy{}, false
	}
	share := int(ref.ContextPerStream())
	return Policy{
		PerStreamContext: share,
		// Hold back a twelfth of the share for the chunk's own answer, which is the ratio the
		// shipped 3090 profile uses (8,192 admitted of 8,874, leaving 682).
		OutputReserve: share / 12,
		MaxChunks:     int(ref.Streams),
		Shape:         shape,
	}, true
}
