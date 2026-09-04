// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package vcontext decides how a request larger than one stream's context is divided across
// streams, and puts it back together.
//
// It sits above the engine's API rather than inside it: the engine serves streams, this decides
// what to put in them. Nothing here imports the engine.
package vcontext

import "fmt"

// Shape is what a request mostly does, which decides whether chunking helps it at all.
//
// This is not a detail. Prefill and decode move in OPPOSITE directions with stream count, so a
// policy that picks chunk count from input size alone picks wrongly for half its traffic.
// Measured on a 3090 with one 131,072-token input split N ways:
//
//	chunks   prefill tok/s   decode tok/s
//	     1          3528.1          78.16
//	    16          3222.8         147.82
//	    32          3057.3         170.02
//	    48          2853.4         164.81
//
// Splitting cost ~19% of ingest and gained ~2x on generation. So a job that reads a lot and emits
// a little is made slower by chunking, and a job that generates substantially is made faster.
type Shape int

const (
	// ShapeUnknown means the caller did not say. Treated as read-heavy, because that is the
	// conservative choice: it splits as little as possible, and the cost of being wrong is a
	// missed speedup rather than a real slowdown.
	ShapeUnknown Shape = iota
	// ShapeReadHeavy is summarise, extract, classify — much in, little out. Prefill-bound.
	// Chunk only as far as needed to fit.
	ShapeReadHeavy
	// ShapeGenerateHeavy produces substantial output, or serves many callers at once.
	// Decode-bound. Chunking earns its keep.
	ShapeGenerateHeavy
)

func (s Shape) String() string {
	switch s {
	case ShapeReadHeavy:
		return "read-heavy"
	case ShapeGenerateHeavy:
		return "generate-heavy"
	default:
		return "unknown"
	}
}

// Policy is the deployment's geometry plus how it should be used.
type Policy struct {
	// PerStreamContext is what one sequence may occupy — the admitted context, not the
	// allocated one. A plan that fills the allocation leaves nothing to generate into.
	PerStreamContext int
	// OutputReserve is held back in every chunk for that chunk's own output. A chunk sized to
	// the whole share cannot answer.
	OutputReserve int
	// MaxChunks is how many streams the deployment can actually run at once. Splitting beyond
	// this does not add parallelism; it adds queueing.
	MaxChunks int
	// Shape is what the request mostly does. See Shape.
	Shape Shape
}

// usable is how many prompt tokens fit in one chunk.
func (p Policy) usable() int {
	n := p.PerStreamContext - p.OutputReserve
	if n < 1 {
		return 0
	}
	return n
}

// Validate reports a policy that cannot produce a working plan.
//
// Checked rather than assumed because the failure is silent otherwise: a reserve larger than the
// context yields chunks with no room to answer, and the model returns nothing for reasons that
// look like a model problem.
func (p Policy) Validate() error {
	if p.PerStreamContext <= 0 {
		return fmt.Errorf("vcontext: per-stream context must be positive, got %d", p.PerStreamContext)
	}
	if p.OutputReserve < 0 {
		return fmt.Errorf("vcontext: output reserve cannot be negative, got %d", p.OutputReserve)
	}
	if p.usable() <= 0 {
		return fmt.Errorf("vcontext: reserving %d of %d tokens leaves no room for input",
			p.OutputReserve, p.PerStreamContext)
	}
	if p.MaxChunks < 1 {
		return fmt.Errorf("vcontext: max chunks must be at least 1, got %d", p.MaxChunks)
	}
	return nil
}
