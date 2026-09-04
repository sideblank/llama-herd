// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"strings"
	"testing"
)

// The pass that would silently corrupt a document. Joining lines inside a code fence changes what
// the code means, and the answer would then be about text that does not exist.
func TestCodeFencesSurviveLineJoining(t *testing.T) {
	src := "Some wrapped prose\nthat should be joined.\n\n" +
		"```\nfor i := range xs {\n    total += i\n}\n```\n\n" +
		"More wrapped prose\nthat should also join."

	got := Canonicalise(src)

	if !strings.Contains(got, "for i := range xs {\n    total += i\n}") {
		t.Errorf("the code block was mangled:\n%s", got)
	}
	if !strings.Contains(got, "wrapped prose that should be joined") {
		t.Error("prose outside the fence was not joined")
	}
	if !strings.Contains(got, "More wrapped prose that should also join") {
		t.Error("prose after the fence was not joined")
	}
}

// Structure markers begin a line for a reason. Joining a bullet or a heading into the line above
// destroys the structure the skeleton pass relies on.
func TestStructureMarkersAreNotJoined(t *testing.T) {
	src := "Introduction text here\n- first bullet\n- second bullet\n\nA paragraph\n# A heading"
	got := Canonicalise(src)
	for _, want := range []string{"\n- first bullet", "\n- second bullet", "\n# A heading"} {
		if !strings.Contains(got, want) {
			t.Errorf("structure marker was joined away: %q missing from\n%s", want, got)
		}
	}
}

// A single space is load-bearing — BPE merges a leading space with the following word — so only
// runs of two or more may collapse.
func TestSingleSpacesSurvive(t *testing.T) {
	src := "the system uses the port"
	if got := Canonicalise(src); got != src {
		t.Errorf("single spaces were altered: %q -> %q", src, got)
	}
}

func TestWhitespacePassesReduceText(t *testing.T) {
	src := "line with trailing spaces   \nand    excessive     inline    spacing\n\n\n\n\nafter"
	got := Canonicalise(src)
	if strings.Contains(got, "   \n") {
		t.Error("trailing spaces survived")
	}
	if strings.Contains(got, "    ") {
		t.Error("runs of spaces survived")
	}
	if strings.Contains(got, "\n\n\n") {
		t.Error("more than two consecutive newlines survived")
	}
}

// The validation rule, made structural: every pass must be measurable on its own, so one that
// breaks a BPE merge can be identified and dropped rather than hidden inside a total.
func TestEachPassIsMeasuredSeparately(t *testing.T) {
	src := "wrapped prose\nacross lines   with    spacing\n\n\n\nand gaps"
	results := MeasureCanon(src, fakeCount)

	if len(results) != len(CanonPasses()) {
		t.Fatalf("measured %d passes, want %d", len(results), len(CanonPasses()))
	}
	for _, r := range results {
		if r.Why == "" {
			t.Errorf("pass %q has no stated reason — a transformation nobody can justify cannot "+
				"be reviewed when it turns out to cost tokens", r.Name)
		}
		if r.BeforeToks == 0 {
			t.Errorf("pass %q was not tokenised", r.Name)
		}
	}
	// Chained, so each pass sees the previous one's output.
	for i := 1; i < len(results); i++ {
		if results[i].BeforeChars != results[i-1].AfterChars {
			t.Errorf("pass %q did not start where %q finished — the passes are not chained",
				results[i].Name, results[i-1].Name)
		}
	}
}

// Canonicalisation must never grow the text. A pass that does has broken a merge, which is exactly
// what the measurement exists to catch.
func TestCanonicalisationNeverGrowsText(t *testing.T) {
	cases := []string{
		"plain prose with nothing to trim",
		"wrapped\nprose\nacross\nmany\nlines",
		"```\ncode\n  indented\n```",
		"- bullet\n- list\n- items",
		"trailing   \nspaces   \neverywhere   ",
		strings.Repeat("mixed content with   spacing\nand wraps\n\n\n", 20),
	}
	for _, src := range cases {
		got := Canonicalise(src)
		if len(got) > len(src) {
			t.Errorf("canonicalisation grew the text by %d chars: %q", len(got)-len(src), src)
		}
	}
}

// Offsets move, so this must happen before indexing. Asserted here because the consequence of
// getting it wrong is invisible: spans would name text they do not cover.
func TestCanonicalisedTextIsTheSourceOfTruth(t *testing.T) {
	raw := "System A   uses port 8080.\nIt is deployed\nin the primary region.\n\n\n\nSystem B."
	canon := Canonicalise(raw)
	if canon == raw {
		t.Skip("nothing to canonicalise in this input")
	}
	chunk := Chunk{Index: 0, Start: 0, End: len(canon), Cut: CutEnd}
	spans := Segment(chunk, canon, 256, fakeCount)
	for _, sp := range spans {
		if _, err := sp.Text(canon); err != nil {
			t.Errorf("span does not validate against the canonicalised source: %v", err)
		}
	}
	// The same index against the raw text is meaningless — offsets have moved.
	ix := &Index{Spans: spans}
	if err := ix.Validate(raw); err == nil && len(canon) != len(raw) {
		t.Log("note: index happened to validate against raw text; offsets still differ in meaning")
	}
}

// Space collapsing turns four spaces of indentation into one. In a whitespace-significant language
// that is a syntax change, not cosmetics — and it is the pass nobody thinks to protect.
func TestIndentationSurvivesInsideFences(t *testing.T) {
	src := "Prose before.\n\n```python\ndef f():\n    if x:\n        return 1\n```\n\nProse after."
	got := Canonicalise(src)
	if !strings.Contains(got, "    if x:") {
		t.Errorf("four-space indent was collapsed:\n%s", got)
	}
	if !strings.Contains(got, "        return 1") {
		t.Errorf("eight-space indent was collapsed:\n%s", got)
	}
}

// Blank lines inside a fence are part of the code's shape and must not be capped away.
func TestBlankLinesInsideFencesSurvive(t *testing.T) {
	src := "Text.\n\n```\nfirst\n\n\n\nsecond\n```\n\nMore."
	got := Canonicalise(src)
	if !strings.Contains(got, "first\n\n\n\nsecond") {
		t.Errorf("newlines inside the fence were capped:\n%s", got)
	}
	if strings.Contains(strings.Split(got, "```")[0], "\n\n\n") {
		t.Error("newlines outside the fence were not capped")
	}
}

// Tilde fences are fences too.
func TestTildeFencesAreProtected(t *testing.T) {
	src := "Text.\n\n~~~\n    indented   code\n~~~\n\nMore."
	if got := Canonicalise(src); !strings.Contains(got, "    indented   code") {
		t.Errorf("a tilde fence was not protected:\n%s", got)
	}
}
