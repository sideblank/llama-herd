// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import "fmt"

// Plan is how one request will be served.
type Plan struct {
	// Chunks is how many pieces the input is divided into. Always at least 1.
	Chunks int
	// ChunkTokens is the TARGET size of each piece, in prompt tokens. Cuts land on boundaries,
	// so real chunks come in at or under it and the remainder accumulates in the last one.
	ChunkTokens int
	// MaxChunkTokens is what a chunk may not exceed: one stream's usable budget.
	//
	// Distinct from ChunkTokens because the target is an average and the limit is a hard edge.
	// Checking a chunk against the average rejects splits that are entirely valid — a final
	// chunk carrying the remainder is normal, and only a chunk that will not fit its stream is
	// an error.
	MaxChunkTokens int
	// Direct is true when the request fits in one stream and is served without splitting,
	// without a store, and without reassembly.
	//
	// This is the common case and the one sub-agent traffic falls into, where the engine is
	// already at its best — many short independent shallow requests measured 564-728 tok/s
	// aggregate. The layer's job there is to get out of the way.
	Direct bool
	// Reason states why this shape was chosen, in a sentence a person can check.
	Reason string
	// Refused is set when the request cannot be served at all, with Reason saying why.
	Refused bool
}

// Planner decides how to divide a request.
//
// It is deliberately free of any engine dependency: it takes a token count and returns a
// decision, so the policy can be tested exhaustively without a GPU. Everything expensive to
// discover about this system is encoded here rather than rediscovered per deployment.
type Planner struct {
	Policy Policy
}

// Plan decides how an input of promptTokens should be served.
//
// The rules, and where each comes from:
//
//   - Fits in one stream: serve directly. No split can beat not splitting, since splitting costs
//     ingest throughput and buys parallelism the request cannot use.
//   - Read-heavy: use the FEWEST chunks that fit. Splitting measured ~19% slower on ingest, and a
//     read-heavy job is ingest-bound, so every extra chunk is a loss.
//   - Generate-heavy: split further, because decode measured ~2x faster spread across streams —
//     and, at constant tokens resident, shallower sequences measured up to 1.9x faster than deeper
//     ones. Depth within a sequence is what costs, not occupancy across the herd.
//   - Never beyond MaxChunks: past the stream count there is no more parallelism to buy, only
//     queueing.
func (p Planner) Plan(promptTokens int) (Plan, error) {
	if err := p.Policy.Validate(); err != nil {
		return Plan{}, err
	}
	if promptTokens < 0 {
		return Plan{}, fmt.Errorf("vcontext: prompt tokens cannot be negative, got %d", promptTokens)
	}

	usable := p.Policy.usable()

	// One stream is enough. Take the cheap path.
	if promptTokens <= usable {
		return Plan{
			Chunks:         1,
			ChunkTokens:    promptTokens,
			MaxChunkTokens: usable,
			Direct:         true,
			Reason: fmt.Sprintf("%d tokens fits one stream's %d usable — served directly, "+
				"no split", promptTokens, usable),
		}, nil
	}

	// The fewest chunks that hold it. This is the floor for every shape.
	minChunks := ceilDiv(promptTokens, usable)

	if minChunks > p.Policy.MaxChunks {
		return Plan{
			Refused: true,
			Reason: fmt.Sprintf("%d tokens needs %d chunks of %d, but only %d streams are "+
				"available — the deployment cannot hold this request",
				promptTokens, minChunks, usable, p.Policy.MaxChunks),
		}, nil
	}

	chunks := minChunks
	reason := fmt.Sprintf("%d tokens over %d chunks — the fewest that fit, because ingest is "+
		"the bound for %s work and splitting costs it", promptTokens, chunks, p.Policy.Shape)

	if p.Policy.Shape == ShapeGenerateHeavy {
		// Spread further: decode is the bound, and shallower sequences decode faster.
		//
		// Doubling is deliberately modest rather than going straight to MaxChunks. The gain
		// flattens — 32 chunks measured 170 tok/s against 48 at 165 — and the far end of the
		// stream range sits next to a capacity cliff whose position moves between machines.
		if want := minChunks * 2; want <= p.Policy.MaxChunks {
			chunks = want
		} else {
			chunks = p.Policy.MaxChunks
		}
		reason = fmt.Sprintf("%d tokens over %d chunks — more than the %d needed to fit, "+
			"because decode is the bound for generate-heavy work and shallower sequences "+
			"decode faster", promptTokens, chunks, minChunks)
	}

	return Plan{
		Chunks:         chunks,
		ChunkTokens:    ceilDiv(promptTokens, chunks),
		MaxChunkTokens: usable,
		Reason:         reason,
	}, nil
}

func ceilDiv(a, b int) int {
	if b <= 0 {
		return 0
	}
	return (a + b - 1) / b
}
