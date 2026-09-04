// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"math"
	"sort"
	"strings"
	"unicode"
)

// LexicalRetriever ranks spans by term overlap with the query.
//
// A floor, not the production path. It runs without a GPU, a database or an embedding model, which
// makes the rest of the pipeline testable end to end today — and it establishes the baseline any
// vector retriever has to beat, because a semantic index that scores worse than word matching is
// not earning its dependency.
//
// Its weakness is the obvious one: it cannot match a question to a passage that answers it in
// different words. That is exactly what embeddings fix, and why Retriever is an interface.
type LexicalRetriever struct {
	// TopicWeight is how much the indexer's topic label counts against the source text. Topics
	// are short and dense, so they need scaling down or they dominate.
	TopicWeight float64
}

func (r LexicalRetriever) Rank(_ context.Context, query, source string, spans []Span) ([]Span, error) {
	q := terms(query)
	if len(q) == 0 {
		// No usable query terms: preserve the index's own ordering rather than inventing one.
		return append([]Span(nil), spans...), nil
	}
	tw := r.TopicWeight
	if tw == 0 {
		tw = 0.5
	}

	// Term frequencies per span, and how many spans each term appears in.
	//
	// Inverse document frequency is what makes this a usable floor rather than a straw man. Without
	// it every query term counts alike, so a word appearing in every span scores as highly as the
	// one that identifies the answer — measured: a query for "port billing node" selected almost
	// every span, because "node" was in all of them and outvoted "billing".
	docs := make([]map[string]int, len(spans))
	df := map[string]int{}
	for i, sp := range spans {
		text, err := sp.Text(source)
		if err != nil {
			return nil, err
		}
		tf := map[string]int{}
		for _, t := range terms(text) {
			tf[t]++
		}
		for _, t := range terms(sp.Topic) {
			tf[t] += 2 // a topic label is short and deliberate, so it counts for more per word
		}
		docs[i] = tf
		for t := range tf {
			df[t]++
		}
	}
	n := float64(len(spans))

	type scored struct {
		span  Span
		score float64
	}
	list := make([]scored, 0, len(spans))
	for i, sp := range spans {
		var score float64
		for _, t := range q {
			f := docs[i][t]
			if f == 0 {
				continue
			}
			// Saturating term frequency: a passage mentioning a term ten times is not ten times
			// more relevant, and rewarding repetition would spend the budget on padding.
			tfw := float64(f) / (float64(f) + 1.5)
			idf := math.Log(1 + (n-float64(df[t])+0.5)/(float64(df[t])+0.5))
			score += tfw * idf
		}
		score *= tw / 0.5 // TopicWeight scales the whole score, preserving the old knob's sense
		score += 0.01 * sp.Weight
		list = append(list, scored{sp, score})
	}
	sort.SliceStable(list, func(i, j int) bool { return list[i].score > list[j].score })

	out := make([]Span, 0, len(list))
	for _, s := range list {
		out = append(out, s.span)
	}
	return out, nil
}

func terms(s string) []string {
	fields := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	out := fields[:0]
	for _, f := range fields {
		if len(f) < 3 || stop[f] {
			continue
		}
		out = append(out, f)
	}
	return out
}

// stop holds words too common to discriminate. Deliberately short: an aggressive list throws away
// terms that matter in a technical document.
var stop = map[string]bool{
	"the": true, "and": true, "for": true, "are": true, "but": true, "not": true,
	"you": true, "all": true, "can": true, "has": true, "was": true, "with": true,
	"that": true, "this": true, "from": true, "have": true, "what": true, "which": true,
	"were": true, "been": true, "does": true, "did": true, "its": true, "into": true,
}
