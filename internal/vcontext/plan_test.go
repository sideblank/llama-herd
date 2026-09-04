// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import "testing"

// The 3090 profile: 48 streams over 425,984 tokens, admitting 8,192 with 682 left to answer in.
func profile3090() Policy {
	return Policy{PerStreamContext: 8874, OutputReserve: 682, MaxChunks: 48}
}

// A request that fits must not be chunked. This is the common case and the one sub-agent traffic
// falls into; adding a pipeline to it would tax the workload the engine is already best at.
func TestRequestThatFitsIsServedDirectly(t *testing.T) {
	p := Planner{Policy: profile3090()}
	for _, tokens := range []int{1, 100, 4096, 8191} {
		got, err := p.Plan(tokens)
		if err != nil {
			t.Fatalf("%d tokens: %v", tokens, err)
		}
		if !got.Direct || got.Chunks != 1 {
			t.Errorf("%d tokens: chunks=%d direct=%v, want 1 and true — a request that fits "+
				"must not be split", tokens, got.Chunks, got.Direct)
		}
	}
}

// Exactly at the boundary is still one stream; one past it is not.
func TestBoundaryIsExact(t *testing.T) {
	pol := profile3090()
	usable := pol.usable() // 8874 - 682 = 8192
	p := Planner{Policy: pol}

	at, _ := p.Plan(usable)
	if !at.Direct {
		t.Errorf("%d tokens should fit one stream exactly", usable)
	}
	over, _ := p.Plan(usable + 1)
	if over.Direct {
		t.Errorf("%d tokens should not fit one stream", usable+1)
	}
	if over.Chunks != 2 {
		t.Errorf("one token over should need 2 chunks, got %d", over.Chunks)
	}
}

// Read-heavy work is ingest-bound, and splitting measured ~19% slower on ingest. So it must use
// the FEWEST chunks that fit — the opposite of what "more parallelism is better" would choose.
func TestReadHeavyUsesFewestChunksThatFit(t *testing.T) {
	pol := profile3090()
	pol.Shape = ShapeReadHeavy
	p := Planner{Policy: pol}

	got, err := p.Plan(131072)
	if err != nil {
		t.Fatal(err)
	}
	want := ceilDiv(131072, pol.usable()) // 16
	if got.Chunks != want {
		t.Errorf("read-heavy 131k: chunks=%d, want %d (the minimum that fits) — extra chunks "+
			"cost ingest throughput and buy nothing for a read-bound job", got.Chunks, want)
	}
}

// Generate-heavy work is decode-bound, where splitting measured ~2x faster, so it should spread
// beyond the minimum.
func TestGenerateHeavySpreadsWider(t *testing.T) {
	pol := profile3090()
	pol.Shape = ShapeGenerateHeavy
	p := Planner{Policy: pol}

	got, err := p.Plan(131072)
	if err != nil {
		t.Fatal(err)
	}
	min := ceilDiv(131072, pol.usable())
	if got.Chunks <= min {
		t.Errorf("generate-heavy: chunks=%d, want more than the %d minimum — decode is the "+
			"bound and shallower sequences decode faster", got.Chunks, min)
	}
	if got.Chunks > pol.MaxChunks {
		t.Errorf("chunks=%d exceeds the %d streams available; past that there is no "+
			"parallelism to buy, only queueing", got.Chunks, pol.MaxChunks)
	}
}

// Unknown shape must behave like read-heavy: split as little as possible. Being wrong that way
// costs a missed speedup; being wrong the other way costs real throughput on a read-bound job.
func TestUnknownShapeIsConservative(t *testing.T) {
	unknown := profile3090()
	read := profile3090()
	read.Shape = ShapeReadHeavy

	a, _ := (Planner{Policy: unknown}).Plan(131072)
	b, _ := (Planner{Policy: read}).Plan(131072)
	if a.Chunks != b.Chunks {
		t.Errorf("unknown shape chose %d chunks, read-heavy chose %d — an undeclared request "+
			"must take the conservative path", a.Chunks, b.Chunks)
	}
}

// A request too large for the whole deployment must be refused, not silently truncated or split
// into chunks that cannot run.
func TestOversizedRequestIsRefused(t *testing.T) {
	pol := profile3090()
	p := Planner{Policy: pol}
	huge := pol.usable()*pol.MaxChunks + 1

	got, err := p.Plan(huge)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Refused {
		t.Errorf("%d tokens exceeds %d streams x %d usable and must be refused, got %d chunks",
			huge, pol.MaxChunks, pol.usable(), got.Chunks)
	}
	if got.Reason == "" {
		t.Error("a refusal must say why")
	}
}

// Every plan must carry a reason. A split with no stated cause cannot be reviewed, and this
// policy is counter-intuitive enough that it will be questioned.
func TestEveryPlanExplainsItself(t *testing.T) {
	for _, shape := range []Shape{ShapeUnknown, ShapeReadHeavy, ShapeGenerateHeavy} {
		pol := profile3090()
		pol.Shape = shape
		for _, tokens := range []int{100, 8192, 131072, 400000} {
			got, err := (Planner{Policy: pol}).Plan(tokens)
			if err != nil {
				t.Fatalf("%s/%d: %v", shape, tokens, err)
			}
			if got.Reason == "" {
				t.Errorf("%s/%d tokens: no reason given", shape, tokens)
			}
		}
	}
}

// Chunks must actually hold the input. An off-by-one here produces chunks that overflow their
// stream and fail at submit time, far from the cause.
func TestChunksHoldTheWholeInput(t *testing.T) {
	for _, shape := range []Shape{ShapeReadHeavy, ShapeGenerateHeavy} {
		pol := profile3090()
		pol.Shape = shape
		for _, tokens := range []int{8193, 20000, 131072, 393216} {
			got, err := (Planner{Policy: pol}).Plan(tokens)
			if err != nil || got.Refused {
				continue
			}
			if got.Chunks*got.ChunkTokens < tokens {
				t.Errorf("%s/%d: %d chunks of %d holds only %d", shape, tokens,
					got.Chunks, got.ChunkTokens, got.Chunks*got.ChunkTokens)
			}
			if got.ChunkTokens > pol.usable() {
				t.Errorf("%s/%d: chunk of %d exceeds the %d usable per stream", shape, tokens,
					got.ChunkTokens, pol.usable())
			}
		}
	}
}

// A policy that cannot work must say so rather than producing plans that fail later.
func TestInvalidPolicyIsRejected(t *testing.T) {
	bad := []Policy{
		{PerStreamContext: 0, MaxChunks: 4},
		{PerStreamContext: 1000, OutputReserve: 1000, MaxChunks: 4},
		{PerStreamContext: 1000, OutputReserve: 100, MaxChunks: 0},
		{PerStreamContext: 1000, OutputReserve: -1, MaxChunks: 4},
	}
	for i, pol := range bad {
		if _, err := (Planner{Policy: pol}).Plan(100); err == nil {
			t.Errorf("policy %d should have been rejected: %+v", i, pol)
		}
	}
}
