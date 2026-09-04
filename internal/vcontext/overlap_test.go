// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"strings"
	"testing"
)

// words is a deterministic stand-in for a tokenizer: one token per whitespace-separated run.
func words(s string) int { return len(strings.Fields(s)) }

func chunkWith(i int, text string, cut CutQuality) Chunk {
	return Chunk{Index: i, Text: text, Tokens: words(text), Cut: cut}
}

func TestNoOverlapAfterADeclarationCut(t *testing.T) {
	chunks := []Chunk{
		chunkWith(0, "func a() {}\n", CutDeclaration),
		chunkWith(1, "func b() {}\n", CutEnd),
	}
	out, err := AddOverlap(chunks, 20, words)
	if err != nil {
		t.Fatal(err)
	}
	if out[1].Context != "" {
		t.Fatal("a declaration boundary severed nothing; borrowing there pays for duplicate prefill and buys nothing")
	}
	if tokens, n := OverlapCost(out); tokens != 0 || n != 0 {
		t.Fatalf("want no cost, got %d tokens across %d chunks", tokens, n)
	}
}

func TestOverlapAfterARaggedCut(t *testing.T) {
	prev := strings.Repeat("alpha beta gamma delta ", 20)
	chunks := []Chunk{
		chunkWith(0, prev, CutHard),
		chunkWith(1, "continues here", CutEnd),
	}
	out, _ := AddOverlap(chunks, 8, words)
	if out[1].Context == "" {
		t.Fatal("a hard cut severed something and is exactly what an overlap window repairs")
	}
	if n := words(out[1].Context); n > 8 {
		t.Fatalf("the window must respect its budget, got %d tokens", n)
	}
	if !strings.HasSuffix(strings.TrimSpace(prev), strings.TrimSpace(out[1].Context)) {
		t.Fatal("the borrowed text must be the predecessor's tail")
	}
}

// The direction bug this guards against.
func TestOverlapIsDecidedByThePrecedingChunksCut(t *testing.T) {
	// chunk 0 ends cleanly; chunk 1 ends badly. Only chunk 2 should borrow.
	chunks := []Chunk{
		chunkWith(0, strings.Repeat("a b c ", 20), CutDeclaration),
		chunkWith(1, strings.Repeat("d e f ", 20), CutHard),
		chunkWith(2, "tail", CutEnd),
	}
	out, _ := AddOverlap(chunks, 6, words)
	if out[1].Context != "" {
		t.Fatal("chunk 1 opens on a declaration boundary — nothing to repair")
	}
	if out[2].Context == "" {
		t.Fatal("chunk 2 opens on a hard cut and must borrow; reading a chunk's own Cut would repair the wrong edge")
	}
}

func TestFirstChunkNeverBorrows(t *testing.T) {
	out, _ := AddOverlap([]Chunk{chunkWith(0, "x y z", CutHard)}, 10, words)
	if out[0].Context != "" {
		t.Fatal("there is nothing before the first chunk")
	}
}

func TestParagraphCutIsTreatedAsClean(t *testing.T) {
	chunks := []Chunk{
		chunkWith(0, strings.Repeat("word ", 40), CutParagraph),
		chunkWith(1, "next", CutEnd),
	}
	out, _ := AddOverlap(chunks, 10, words)
	if out[1].Context != "" {
		t.Fatal("a blank line is a boundary the author chose; that is why it ranks above a sentence break")
	}
}

func TestZeroWindowDisablesOverlap(t *testing.T) {
	chunks := []Chunk{chunkWith(0, "a b c", CutHard), chunkWith(1, "d", CutEnd)}
	out, _ := AddOverlap(chunks, 0, words)
	if out[1].Context != "" {
		t.Fatal("a zero window means no borrowing")
	}
}

func TestPromptPutsContextFirst(t *testing.T) {
	o := Overlapped{Chunk: chunkWith(1, "BODY", CutEnd), Context: "TAIL "}
	if o.Prompt() != "TAIL BODY" {
		t.Fatalf("got %q", o.Prompt())
	}
	bare := Overlapped{Chunk: chunkWith(0, "BODY", CutEnd)}
	if bare.Prompt() != "BODY" {
		t.Fatal("no context, no prefix")
	}
}

func TestOverlapCostIsReported(t *testing.T) {
	chunks := []Chunk{
		chunkWith(0, strings.Repeat("a b ", 30), CutHard),
		chunkWith(1, strings.Repeat("c d ", 30), CutHard),
		chunkWith(2, "end", CutEnd),
	}
	out, _ := AddOverlap(chunks, 6, words)
	tokens, n := OverlapCost(out)
	if n != 2 {
		t.Fatalf("two chunks follow a ragged cut, got %d", n)
	}
	if tokens == 0 {
		t.Fatal("the duplicated prefill must be visible rather than inferred — at 48 streams it is the difference between a repair and a tax")
	}
}

func TestLastTokensStartsOnAWordBoundary(t *testing.T) {
	got := lastTokens("alpha beta gamma delta", 2, words)
	if strings.HasPrefix(got, "amma") || strings.HasPrefix(got, "elta") {
		t.Fatalf("opening mid-word gives the model a fragment that tokenises differently from the word it came from: %q", got)
	}
	if words(got) > 2 {
		t.Fatalf("over budget: %q", got)
	}
}

func TestAddOverlapNeedsACounter(t *testing.T) {
	if _, err := AddOverlap([]Chunk{chunkWith(0, "x", CutEnd)}, 5, nil); err == nil {
		t.Fatal("estimating the window size is how it ends up the wrong size silently")
	}
}
