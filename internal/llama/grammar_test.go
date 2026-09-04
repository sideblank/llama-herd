//go:build llamaabi

// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

import "testing"

// A grammar must be rejected loudly at chain construction, not silently ignored.
//
// Silently dropping it is the dangerous failure: generation proceeds, output looks plausible, and
// a merge downstream discovers the shape is wrong on some later response rather than at the point
// the mistake was made.
func TestBadGrammarIsRejected(t *testing.T) {
	if _, err := NewSampler(SamplingParams{Temperature: 0.8, Grammar: "this is not gbnf ((("},
		32000, nil); err == nil {
		t.Error("a malformed grammar with no vocab must error")
	}
}

// A grammar needs the vocabulary it constrains; asking for one without it is a caller mistake and
// must be named as such.
func TestGrammarRequiresVocab(t *testing.T) {
	_, err := NewSampler(SamplingParams{Temperature: 0.8, Grammar: `root ::= "{}"`}, 32000, nil)
	if err == nil {
		t.Fatal("expected an error when a grammar is given without a vocabulary")
	}
}
