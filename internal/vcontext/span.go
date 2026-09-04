// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"fmt"
	"sort"
)

// Span is a region of the source, identified by offset rather than copied.
//
// Offsets rather than text, deliberately. The final answer must be generated from the ORIGINAL
// bytes, not from anything written about them — a summary loses exactly the detail a question
// turns on, and no care in reassembly puts it back. An index of pointers keeps the source
// authoritative and lets selection load verbatim.
type Span struct {
	// Start and End are byte offsets into the original input, half-open.
	Start, End int
	// Chunk is which chunk produced this, for attribution and for re-reading.
	Chunk int
	// Topic is what the indexing pass says this region is about. Used for ranking, never
	// substituted for the source text.
	Topic string
	// Weight is the indexer's confidence that this span is substantive rather than boilerplate.
	Weight float64
}

// Len is the span's size in bytes.
func (s Span) Len() int { return s.End - s.Start }

// Text returns the span's original bytes from the source it indexes.
func (s Span) Text(source string) (string, error) {
	if s.Start < 0 || s.End > len(source) || s.Start > s.End {
		return "", fmt.Errorf("vcontext: span [%d,%d) is outside a source of %d bytes — the "+
			"index does not belong to this input", s.Start, s.End, len(source))
	}
	return source[s.Start:s.End], nil
}

// Index is every span known about one input, in source order.
type Index struct {
	// Skeleton is the global outline: what this document is, its structure, the entities
	// running through it. Built from a downsampled read before any chunk is processed, and
	// carried into every context so nothing is ever reasoned about in isolation.
	Skeleton string
	// Spans are in source order, which is the order they must be assembled in — reading order
	// is what makes an answer coherent.
	Spans []Span
}

// Sort puts spans in source order and merges any that overlap.
//
// Overlap matters: two indexers describing the same region would otherwise cause that text to be
// loaded twice, spending the answer budget on a duplicate and pushing out something else.
func (ix *Index) Sort() {
	sort.SliceStable(ix.Spans, func(i, j int) bool {
		if ix.Spans[i].Start != ix.Spans[j].Start {
			return ix.Spans[i].Start < ix.Spans[j].Start
		}
		return ix.Spans[i].End < ix.Spans[j].End
	})
	merged := ix.Spans[:0]
	for _, s := range ix.Spans {
		if n := len(merged); n > 0 && s.Start <= merged[n-1].End {
			if s.End > merged[n-1].End {
				merged[n-1].End = s.End
			}
			if s.Weight > merged[n-1].Weight {
				merged[n-1].Weight = s.Weight
			}
			continue
		}
		merged = append(merged, s)
	}
	ix.Spans = merged
}

// Validate checks that the index actually describes the source it will be used against.
//
// An index built from a different input produces an answer assembled from the wrong bytes, which
// reads as a confident, coherent, wrong answer — the failure mode this design most needs to avoid.
func (ix *Index) Validate(source string) error {
	for i, s := range ix.Spans {
		if _, err := s.Text(source); err != nil {
			return fmt.Errorf("span %d: %w", i, err)
		}
		if s.Len() <= 0 {
			return fmt.Errorf("vcontext: span %d is empty", i)
		}
	}
	return nil
}

// Covers reports the fraction of the source that any span points at. A low number means the
// indexing pass skipped most of the input, and an answer drawn from it would be answering about a
// document the caller did not send.
func (ix *Index) Covers(source string) float64 {
	if len(source) == 0 {
		return 0
	}
	total := 0
	for _, s := range ix.Spans {
		total += s.Len()
	}
	if total > len(source) {
		total = len(source)
	}
	return float64(total) / float64(len(source))
}
