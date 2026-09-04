// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"strings"
	"testing"
)

func chunksAt(cuts ...CutQuality) []Chunk {
	var out []Chunk
	pos := 0
	for i, c := range cuts {
		out = append(out, Chunk{Index: i, Start: pos, End: pos + 1000, Cut: c})
		pos += 1000
	}
	return out
}

// The spare capacity is real and must be computed from the deployment, not assumed.
func TestBudgetLeavesSpareForBridges(t *testing.T) {
	// 48 streams, 256k of input at 8k chunks = 32 chunks.
	b, err := Budget(48, 32)
	if err != nil {
		t.Fatal(err)
	}
	if b.Spare != 16 {
		t.Errorf("spare = %d, want 16 — this is the capacity bridges ride for free", b.Spare)
	}
}

func TestBudgetRefusesOversubscription(t *testing.T) {
	if _, err := Budget(48, 49); err == nil {
		t.Error("49 chunks must not fit 48 streams")
	}
}

// The splitter already knows which cuts were bad. Bridges must go to the worst ones first, because
// a paragraph break probably severed nothing and a mid-word break certainly did.
func TestWorstCutsAreBridgedFirst(t *testing.T) {
	chunks := chunksAt(CutParagraph, CutHard, CutParagraph, CutWord, CutSentence, CutEnd)
	b, _ := Budget(48, len(chunks))
	b.Spare = 2 // only two bridges affordable

	got := PlanBridges(chunks, b, 100, nil)
	if len(got) != 2 {
		t.Fatalf("planned %d bridges against a spare of 2", len(got))
	}
	if got[0].Left != 1 {
		t.Errorf("first bridge repairs chunk %d; the hard cut at chunk 1 should outrank it",
			got[0].Left)
	}
	if got[1].Left != 3 {
		t.Errorf("second bridge repairs chunk %d; the word cut at chunk 3 should be next",
			got[1].Left)
	}
}

// Bridges must never exceed the spare, or they spill into a second wave and cost a full pass —
// the opposite of the point.
func TestBridgesNeverExceedSpareCapacity(t *testing.T) {
	chunks := chunksAt(CutHard, CutHard, CutHard, CutHard, CutHard, CutHard, CutHard, CutEnd)
	for _, spare := range []int{0, 1, 3, 100} {
		b := StreamBudget{Total: 48, Chunks: len(chunks), Spare: spare}
		got := PlanBridges(chunks, b, 100, nil)
		if len(got) > spare {
			t.Errorf("spare %d: planned %d bridges", spare, len(got))
		}
	}
}

// An adjacent bridge must actually span the cut — text from both sides, or it repairs nothing.
func TestAdjacentBridgeSpansBothSides(t *testing.T) {
	source := strings.Repeat("left side content. ", 60) + strings.Repeat("right side content. ", 60)
	mid := len(source) / 2
	chunks := []Chunk{
		{Index: 0, Start: 0, End: mid, Cut: CutWord},
		{Index: 1, Start: mid, End: len(source), Cut: CutEnd},
	}
	b := StreamBudget{Total: 48, Chunks: 2, Spare: 4}
	got := PlanBridges(chunks, b, 200, nil)
	if len(got) != 1 {
		t.Fatalf("expected one bridge, got %d", len(got))
	}
	text, err := got[0].Text(source)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Start >= mid || got[0].End <= mid {
		t.Errorf("bridge [%d,%d) does not straddle the cut at %d", got[0].Start, got[0].End, mid)
	}
	if !strings.Contains(text, "left side") || !strings.Contains(text, "right side") {
		t.Error("bridge text does not contain material from both sides of the cut")
	}
}

// A long-range pair joins two distant regions, and the join must be marked — otherwise the model
// reads them as contiguous and reasons about a document that does not exist.
func TestLongRangePairsAreMarkedAsDiscontiguous(t *testing.T) {
	source := strings.Repeat("x", 6000)
	chunks := chunksAt(CutParagraph, CutParagraph, CutParagraph, CutParagraph, CutEnd)
	b := StreamBudget{Total: 48, Chunks: 5, Spare: 10}

	related := func(a, c Chunk) float64 {
		if a.Index == 0 && c.Index == 3 {
			return 0.9
		}
		return 0
	}
	got := PlanBridges(chunks, b, 100, related)

	var far *Bridge
	for i := range got {
		if !got[i].Adjacent {
			far = &got[i]
			break
		}
	}
	if far == nil {
		t.Fatal("no long-range bridge planned despite a related pair")
	}
	if far.Left != 0 || far.Right != 3 {
		t.Errorf("paired chunks %d and %d, want 0 and 3", far.Left, far.Right)
	}
	text, err := far.Text(source)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(text, "not contiguous") {
		t.Error("distant halves were joined with no marker")
	}
}

// Blind spots must be nameable. A severed boundary left unrepaired is the difference between "the
// document does not say" and "we did not look", and a system that cannot report that cannot be
// trusted to say a document is silent on something.
func TestUnrepairedCutsAreReported(t *testing.T) {
	chunks := chunksAt(CutHard, CutHard, CutWord, CutParagraph, CutEnd)
	b := StreamBudget{Total: 48, Chunks: 5, Spare: 1}
	planned := PlanBridges(chunks, b, 100, nil)

	left := UnbridgedCuts(chunks, planned)
	if len(left) == 0 {
		t.Fatal("three severed boundaries and one bridge, but nothing reported unrepaired")
	}
	if left[0].Cut < CutWord {
		t.Errorf("worst unrepaired cut is %s; clean cuts should not be reported", left[0].Cut)
	}
	for _, c := range left {
		if c.Cut < CutWord {
			t.Errorf("chunk %d cut %s reported as a blind spot", c.Index, c.Cut)
		}
	}
}

// A clean split has no blind spots to report.
func TestCleanCutsAreNotReportedAsBlindSpots(t *testing.T) {
	chunks := chunksAt(CutParagraph, CutParagraph, CutSentence, CutEnd)
	if got := UnbridgedCuts(chunks, nil); len(got) != 0 {
		t.Errorf("reported %d blind spots for a cleanly split document", len(got))
	}
}
