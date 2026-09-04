// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"strings"
	"testing"
)

// A stand-in counter. Four characters per token is roughly English prose, and being approximate
// is fine here because the tests assert relationships rather than absolute sizes.
func fakeCount(s string) int { return (len(s) + 3) / 4 }

func para(n int) string {
	p := "Ocean currents move heat from the equator toward the poles. " +
		"Wind drives the surface and density drives the deep return. "
	return strings.Repeat(p, n)
}

// Nothing may be lost. A splitter that drops input produces answers about a document the caller
// did not send, and nothing downstream can detect it.
func TestSplitPreservesEveryCharacter(t *testing.T) {
	text := para(200)
	pol := Policy{PerStreamContext: 1000, OutputReserve: 200, MaxChunks: 48}
	plan, err := (Planner{Policy: pol}).Plan(fakeCount(text))
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := Split(text, plan, fakeCount)
	if err != nil {
		t.Fatal(err)
	}

	var rebuilt strings.Builder
	for _, c := range chunks {
		rebuilt.WriteString(c.Text)
	}
	// Boundary whitespace is trimmed when cutting, so compare with it removed.
	got := strings.Join(strings.Fields(rebuilt.String()), " ")
	want := strings.Join(strings.Fields(text), " ")
	if got != want {
		t.Errorf("split lost or altered content: %d chars in, %d out", len(want), len(got))
	}
}

// No chunk may exceed what a stream will admit, or it is refused at submit time — far from here.
func TestNoChunkExceedsTheStreamShare(t *testing.T) {
	text := para(300)
	pol := Policy{PerStreamContext: 1000, OutputReserve: 200, MaxChunks: 48}
	plan, _ := (Planner{Policy: pol}).Plan(fakeCount(text))
	chunks, err := Split(text, plan, fakeCount)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range chunks {
		if c.Tokens > pol.usable() {
			t.Errorf("chunk %d holds %d tokens against %d usable", c.Index, c.Tokens, pol.usable())
		}
	}
}

// Cuts should land on sentence or paragraph boundaries rather than mid-clause, because a chunk
// ending in the middle of a sentence produces an answer about a fragment.
func TestCutsPreferCleanBoundaries(t *testing.T) {
	text := para(200)
	pol := Policy{PerStreamContext: 1000, OutputReserve: 200, MaxChunks: 48}
	plan, _ := (Planner{Policy: pol}).Plan(fakeCount(text))
	chunks, _ := Split(text, plan, fakeCount)

	clean := 0
	for _, c := range chunks[:len(chunks)-1] { // the last chunk ends where the input does
		if strings.HasSuffix(strings.TrimSpace(c.Text), ".") {
			clean++
		}
	}
	if want := (len(chunks) - 1) / 2; clean < want {
		t.Errorf("only %d of %d cuts landed on a sentence end; want at least %d",
			clean, len(chunks)-1, want)
	}
}

// A direct plan must pass the text through untouched — no splitting machinery on the common path.
func TestDirectPlanIsPassthrough(t *testing.T) {
	text := "a short question"
	plan, _ := (Planner{Policy: Policy{PerStreamContext: 1000, OutputReserve: 200, MaxChunks: 4}}).
		Plan(fakeCount(text))
	if !plan.Direct {
		t.Fatal("expected a direct plan")
	}
	chunks, err := Split(text, plan, fakeCount)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 || chunks[0].Text != text {
		t.Errorf("direct plan altered the input: %d chunks, %q", len(chunks), chunks[0].Text)
	}
}

// Splitting without a counter must fail loudly. Estimating produces chunks that overflow their
// stream, and the failure appears at submit time as a rejected request.
func TestSplitRequiresACounter(t *testing.T) {
	if _, err := Split("text", Plan{Chunks: 2, ChunkTokens: 10}, nil); err == nil {
		t.Error("expected a refusal without a token counter")
	}
}

// A refused plan must not be split anyway.
func TestRefusedPlanCannotBeSplit(t *testing.T) {
	_, err := Split("text", Plan{Refused: true, Reason: "too large"}, fakeCount)
	if err == nil {
		t.Error("expected splitting a refused plan to fail")
	}
}

// Progress must be made even when a single token exceeds the target, or the splitter loops.
func TestPathologicalInputTerminates(t *testing.T) {
	text := strings.Repeat("x", 10000) // no boundaries anywhere
	pol := Policy{PerStreamContext: 100, OutputReserve: 20, MaxChunks: 48}
	plan, err := (Planner{Policy: pol}).Plan(fakeCount(text))
	if err != nil {
		t.Fatal(err)
	}
	if plan.Refused {
		return // legitimately too large; nothing to split
	}
	chunks, err := Split(text, plan, fakeCount)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) == 0 {
		t.Error("no chunks produced")
	}
}

// Boundary-preferring cuts make earlier chunks smaller than the average, so the remainder piles
// into the last one. A final chunk above the AVERAGE is ordinary; only one that will not fit its
// STREAM is a fault. Checking against the average rejects valid splits — which it did, on the
// first end-to-end run.
func TestFinalChunkMayExceedTheAverage(t *testing.T) {
	pol := Policy{PerStreamContext: 220, OutputReserve: 18, MaxChunks: 6}
	text := para(40)
	plan, err := (Planner{Policy: pol}).Plan(fakeCount(text))
	if err != nil {
		t.Fatal(err)
	}
	if plan.Refused {
		t.Skip("input too large for this policy")
	}
	chunks, err := Split(text, plan, fakeCount)
	if err != nil {
		t.Fatalf("a valid split was refused: %v", err)
	}
	last := chunks[len(chunks)-1]
	if last.Tokens > plan.MaxChunkTokens {
		t.Errorf("final chunk of %d exceeds the %d stream limit", last.Tokens, plan.MaxChunkTokens)
	}
	if last.Tokens <= plan.ChunkTokens {
		t.Logf("final chunk %d did not exceed the %d average — the case is untested here",
			last.Tokens, plan.ChunkTokens)
	}
}

// The splitter may return more chunks than the plan predicted, and must never return one that
// does not fit a stream. Boundary-preferring cuts guarantee the shortfall accumulates, so the
// predicted count is a floor rather than a promise.
func TestSplitMayExceedThePredictedCountButNeverTheLimit(t *testing.T) {
	for _, n := range []int{10, 40, 120, 300} {
		pol := Policy{PerStreamContext: 220, OutputReserve: 18, MaxChunks: 64}
		text := para(n)
		plan, err := (Planner{Policy: pol}).Plan(fakeCount(text))
		if err != nil || plan.Refused {
			continue
		}
		chunks, err := Split(text, plan, fakeCount)
		if err != nil {
			t.Fatalf("para(%d): valid split refused: %v", n, err)
		}
		for _, c := range chunks {
			if c.Tokens > plan.MaxChunkTokens {
				t.Errorf("para(%d): chunk %d holds %d against a %d stream limit",
					n, c.Index, c.Tokens, plan.MaxChunkTokens)
			}
		}
		if len(chunks) < plan.Chunks {
			t.Errorf("para(%d): produced %d chunks, fewer than the %d predicted",
				n, len(chunks), plan.Chunks)
		}
	}
}

// A tail fragment is worse than a slightly larger neighbour: it costs a stream and answers
// confidently from too little context. Measured end to end — a 21-token tail produced a digest
// with the wrong value, because the sentence naming it stayed in the previous chunk.
func TestRuntTailIsFoldedBack(t *testing.T) {
	pol := Policy{PerStreamContext: 220, OutputReserve: 18, MaxChunks: 64}
	// Lengths chosen to leave a small remainder after boundary-preferring cuts.
	for _, n := range []int{7, 11, 13, 17, 23, 31} {
		text := para(n)
		plan, err := (Planner{Policy: pol}).Plan(fakeCount(text))
		if err != nil || plan.Refused {
			continue
		}
		chunks, err := Split(text, plan, fakeCount)
		if err != nil {
			t.Fatalf("para(%d): %v", n, err)
		}
		if len(chunks) < 2 {
			continue
		}
		last := chunks[len(chunks)-1]
		if last.Tokens < plan.ChunkTokens/4 {
			t.Errorf("para(%d): tail of %d tokens survived against a %d target — runts must be "+
				"folded into their predecessor", n, last.Tokens, plan.ChunkTokens)
		}
		// Folding must never create an oversized chunk.
		for _, c := range chunks {
			if c.Tokens > plan.MaxChunkTokens {
				t.Errorf("para(%d): folding produced a chunk of %d against a %d limit",
					n, c.Tokens, plan.MaxChunkTokens)
			}
		}
	}
}
