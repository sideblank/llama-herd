// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"fmt"
	"strings"
)

// Overlapped is a chunk with a tail of its predecessor prepended.
type Overlapped struct {
	Chunk
	// Context is the borrowed text from the previous chunk, empty when none was needed.
	Context string
	// ContextTokens is what the borrowed text costs.
	ContextTokens int
}

// Prompt is the text to send: the borrowed context followed by the chunk's own content.
func (o Overlapped) Prompt() string {
	if o.Context == "" {
		return o.Text
	}
	return o.Context + o.Text
}

// AddOverlap prepends a tail of the previous chunk to chunks whose leading edge was cut badly.
//
// ⛔ Conditional, not blanket. An overlap window exists to repair a cut that severed something, and
// a cut that landed on a declaration boundary severed nothing — so borrowing there pays for
// duplicate prefill and buys nothing.
//
// The cost is why this matters: at 48 streams a 300-token unconditional window is ~14,400 tokens of
// duplicated ingest, against a budget of 8,874 per stream. Spending that only where a cut was
// actually ragged is the difference between a repair and a tax.
//
// A chunk borrows based on how ITS OWN leading edge was cut, which is the trailing cut of the chunk
// before it — the boundary they share. Reading the chunk's own Cut instead would repair the wrong
// edge, and every chunk would be compensating for a cut at the far end from the damage.
func AddOverlap(chunks []Chunk, tokens int, count CountTokens) ([]Overlapped, error) {
	if count == nil {
		return nil, fmt.Errorf("vcontext: AddOverlap needs a token counter")
	}
	out := make([]Overlapped, 0, len(chunks))
	for i, c := range chunks {
		o := Overlapped{Chunk: c}
		if i > 0 && tokens > 0 && needsOverlap(chunks[i-1].Cut) {
			tail := lastTokens(chunks[i-1].Text, tokens, count)
			if strings.TrimSpace(tail) != "" {
				o.Context = tail
				o.ContextTokens = count(tail)
			}
		}
		out = append(out, o)
	}
	return out, nil
}

// needsOverlap reports whether a boundary of this quality severed something worth repairing.
//
// CutEnd and CutDeclaration did not. CutParagraph is the interesting case and is treated as
// needing none: a blank line in prose is a deliberate boundary by the author, and the whole reason
// it ranks above a sentence break is that it is unlikely to have broken a relationship.
func needsOverlap(prev CutQuality) bool {
	return prev > CutParagraph
}

// OverlapCost reports the total borrowed tokens and how many chunks borrowed, so the tax is
// visible rather than inferred.
func OverlapCost(o []Overlapped) (tokens, chunks int) {
	for _, c := range o {
		if c.ContextTokens > 0 {
			tokens += c.ContextTokens
			chunks++
		}
	}
	return tokens, chunks
}

// lastTokens returns the trailing run of s costing at most n tokens, starting at a word boundary.
//
// Binary search on the offset rather than a character estimate, for the same reason the splitter
// does: tokens per character varies with content, so a fixed divisor produces a window that is the
// wrong size in a way nothing reports.
func lastTokens(s string, n int, count CountTokens) string {
	if n <= 0 || s == "" {
		return ""
	}
	if count(s) <= n {
		return s
	}
	lo, hi := 0, len(s)
	for lo < hi {
		mid := (lo + hi) / 2
		if count(s[mid:]) <= n {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	tail := s[lo:]
	// Start on a word boundary: opening mid-word gives the model a fragment that tokenises
	// differently from the word it came from.
	if i := strings.IndexAny(tail, " \n\t"); i >= 0 && i < len(tail)-1 {
		tail = tail[i+1:]
	}
	return tail
}
