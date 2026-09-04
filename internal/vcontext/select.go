// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"fmt"
	"sort"
)

// Retriever ranks indexed spans against a query.
//
// This is the interface a real store implements, and it is where the design needs a database
// rather than a map. The workload is not chunk hand-off — it is search over the span index: 256k
// of input is hundreds of spans, ranked by similarity to a query, re-ranked on a second round,
// and ideally persisted so a document read once is not read again at ~78 seconds a time.
//
// Vector similarity, full-text, or both. Lexical is implemented here so the pipeline is testable
// and runnable without a GPU or a database; it is a floor, not the intended production path.
type Retriever interface {
	// Rank returns spans ordered by relevance to the query, best first. It may return fewer
	// than it was given, but must not invent spans.
	Rank(ctx context.Context, query string, source string, spans []Span) ([]Span, error)
}

// Selection is what will be assembled into the answer context.
type Selection struct {
	// Spans are in SOURCE order, not relevance order. Relevance decides what is included;
	// reading order decides how it is assembled, because an answer built from passages shuffled
	// out of sequence reads as incoherent however well each passage was chosen.
	Spans []Span
	// Tokens is what the selected text costs.
	Tokens int
	// Coverage is the fraction of the source that made it in. Low coverage on an aggregate
	// question is the signal that the wrong path was taken.
	Coverage float64
	// Truncated is set when relevant spans were left out for want of budget. The caller needs
	// this: it is the difference between "the document does not say" and "we did not look".
	Truncated bool
}

// Selector fills one stream's context with the most relevant source text.
type Selector struct {
	Retriever Retriever
	Count     CountTokens
	// Budget is how many tokens the answer context may spend on source, after the skeleton,
	// the query, and room to generate.
	Budget int
}

// Select ranks spans, takes the best until the budget is spent, and returns them in source order.
//
// What it never does is substitute anything for the source. The spans it returns are pointers
// into the original bytes, and the answer pass reads those bytes. A summary would lose the detail
// the question turns on, which is the failure this whole path exists to avoid.
func (s Selector) Select(ctx context.Context, query string, ix *Index, source string) (Selection, error) {
	if s.Count == nil {
		return Selection{}, fmt.Errorf("vcontext: a token counter is required")
	}
	if s.Budget <= 0 {
		return Selection{}, fmt.Errorf("vcontext: a positive budget is required, got %d", s.Budget)
	}
	if err := ix.Validate(source); err != nil {
		return Selection{}, err
	}

	ranked, err := s.Retriever.Rank(ctx, query, source, ix.Spans)
	if err != nil {
		return Selection{}, err
	}

	var taken []Span
	spent, truncated := 0, false
	for i, sp := range ranked {
		text, err := sp.Text(source)
		if err != nil {
			return Selection{}, err
		}
		cost := s.Count(text)
		if spent+cost > s.Budget {
			// The best-ranked span not fitting is fatal, not a reason to look further down.
			//
			// Skipping it and taking something smaller means answering a question about
			// termination from a passage about shipping — the budget silently swaps the
			// evidence for filler and the answer is confident and wrong. Filling leftover
			// budget with lower-ranked spans is fine; substituting for the primary evidence
			// is not.
			if i == 0 {
				return Selection{}, fmt.Errorf("vcontext: the most relevant passage needs %d "+
					"tokens against a budget of %d — answering from anything else would be "+
					"answering a different question", cost, s.Budget)
			}
			truncated = true
			continue
		}
		taken = append(taken, sp)
		spent += cost
	}

	// Nothing fitting is a failure, not an empty result.
	//
	// An answer context containing a skeleton and no source would be answered from the model's
	// own knowledge and would read as confident and grounded. That is the worst outcome this
	// design can produce, so it is refused here rather than emitted.
	if len(taken) == 0 && len(ranked) > 0 {
		smallest := ranked[0].Len()
		for _, sp := range ranked {
			if sp.Len() < smallest {
				smallest = sp.Len()
			}
		}
		return Selection{}, fmt.Errorf("vcontext: no span fits a %d-token budget (smallest is "+
			"~%d bytes) — the budget is too small for this index, and answering without source "+
			"would be a fabrication", s.Budget, smallest)
	}

	// Back to reading order for assembly.
	sort.SliceStable(taken, func(i, j int) bool { return taken[i].Start < taken[j].Start })

	covered := 0
	for _, sp := range taken {
		covered += sp.Len()
	}
	cov := 0.0
	if len(source) > 0 {
		cov = float64(covered) / float64(len(source))
	}
	return Selection{Spans: taken, Tokens: spent, Coverage: cov, Truncated: truncated}, nil
}

// Assemble builds the answer context: the skeleton, then the selected source in reading order.
//
// Gaps are marked rather than hidden. A model handed two passages spliced together with no sign
// that anything was omitted will read them as contiguous and reason about a document that does not
// exist; told there is a gap, it can say so.
func (sel Selection) Assemble(ix *Index, source string) (string, error) {
	var b []byte
	if ix.Skeleton != "" {
		b = append(b, "# Document overview\n\n"...)
		b = append(b, ix.Skeleton...)
		b = append(b, "\n\n# Relevant passages, in document order\n\n"...)
	}
	prev := -1
	for _, sp := range sel.Spans {
		if prev >= 0 && sp.Start > prev {
			b = append(b, fmt.Sprintf("\n\n[... %d characters omitted ...]\n\n", sp.Start-prev)...)
		}
		text, err := sp.Text(source)
		if err != nil {
			return "", err
		}
		b = append(b, text...)
		prev = sp.End
	}
	return string(b), nil
}
