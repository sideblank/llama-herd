// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"strings"
	"testing"
)

func oneChunk(text string) Chunk {
	return Chunk{Index: 0, Text: text, Start: 0, End: len(text), Cut: CutEnd}
}

// The property everything downstream rests on: a span's offsets must name the text it claims. An
// off-by-one assembles the answer from the wrong region and nothing detects it.
func TestSpanOffsetsAreExact(t *testing.T) {
	src := "First paragraph about ports.\n\nSecond paragraph about timeouts.\n\nThird about retries."
	spans := Segment(oneChunk(src), src, 256, fakeCount)
	if len(spans) != 3 {
		t.Fatalf("got %d spans, want 3", len(spans))
	}
	for i, want := range []string{"ports", "timeouts", "retries"} {
		got, err := spans[i].Text(src)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(got, want) {
			t.Errorf("span %d = %q, expected it to contain %q", i, got, want)
		}
	}
}

// Offsets must be absolute into the source, not relative to the chunk — a chunk starting midway
// through a document would otherwise index the wrong region entirely.
func TestOffsetsAreAbsoluteNotChunkRelative(t *testing.T) {
	src := "IGNORE THIS PREFIX.\n\nThe termination clause requires ninety days notice."
	chunk := Chunk{Index: 1, Start: 21, End: len(src)}
	spans := Segment(chunk, src, 256, fakeCount)
	if len(spans) == 0 {
		t.Fatal("no spans")
	}
	got, err := spans[0].Text(src)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(got, "IGNORE") {
		t.Errorf("span read from before the chunk start: %q", got)
	}
	if !strings.Contains(got, "ninety days") {
		t.Errorf("span = %q", got)
	}
}

// A paragraph too large to select usefully falls back to sentences, so choosing one span does not
// spend the answer budget on surrounding material nobody asked for.
func TestLargeParagraphsFallBackToSentences(t *testing.T) {
	long := strings.Repeat("This is a sentence about ocean currents. ", 40)
	spans := Segment(oneChunk(long), long, 20, fakeCount)
	if len(spans) < 5 {
		t.Fatalf("a %d-token paragraph produced %d spans against a 20-token limit",
			fakeCount(long), len(spans))
	}
	for i, sp := range spans {
		text, _ := sp.Text(long)
		if !strings.HasSuffix(strings.TrimSpace(text), ".") {
			t.Errorf("span %d does not end at a sentence terminator: %q", i, text)
		}
	}
}

// A decimal point is not a sentence boundary. Splitting there produces spans that end mid-number,
// which reads as truncated to whatever consumes them.
func TestDecimalsDoNotSplitSentences(t *testing.T) {
	src := "The threshold is 3.5 seconds. The retry limit is 2.75 minutes."
	spans := Segment(oneChunk(src), src, 5, fakeCount)
	for _, sp := range spans {
		text, _ := sp.Text(src)
		if strings.HasSuffix(strings.TrimSpace(text), "3.") || strings.HasSuffix(strings.TrimSpace(text), "2.") {
			t.Errorf("split inside a decimal: %q", text)
		}
	}
}

// Every span must validate against the source it indexes, or assembly reads the wrong bytes.
func TestSegmentedIndexValidates(t *testing.T) {
	src, ix := buildDoc()
	chunks := []Chunk{{Index: 0, Start: 0, End: len(src), Cut: CutEnd}}
	got := SegmentAll(chunks, src, 64, fakeCount)
	if err := got.Validate(src); err != nil {
		t.Fatalf("segmented index does not validate: %v", err)
	}
	if len(got.Spans) == 0 {
		t.Fatal("no spans produced")
	}
	_ = ix
}

// Spans must arrive in document order and must not overlap, or selection double-counts budget.
func TestSpansAreOrderedAndDisjoint(t *testing.T) {
	src := strings.Repeat("Alpha beta gamma delta. ", 30) + "\n\n" +
		strings.Repeat("Epsilon zeta eta theta. ", 30)
	chunks := []Chunk{
		{Index: 0, Start: 0, End: 360},
		{Index: 1, Start: 360, End: len(src)},
	}
	ix := SegmentAll(chunks, src, 30, fakeCount)
	for i := 1; i < len(ix.Spans); i++ {
		if ix.Spans[i].Start < ix.Spans[i-1].End {
			t.Errorf("spans %d and %d overlap: [%d,%d) then [%d,%d)",
				i-1, i, ix.Spans[i-1].Start, ix.Spans[i-1].End, ix.Spans[i].Start, ix.Spans[i].End)
		}
	}
}

func TestSegmentRefusesAMismatchedChunk(t *testing.T) {
	src := "short"
	if got := Segment(Chunk{Start: 0, End: 9999}, src, 64, fakeCount); got != nil {
		t.Error("a chunk pointing past the source produced spans")
	}
}
