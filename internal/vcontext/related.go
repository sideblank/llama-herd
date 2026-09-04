// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"math"
	"strings"
)

// RareTermRelated scores how likely two chunks are to discuss the same thing, from the terms they
// share weighted by how rare those terms are across the document.
//
// This is the floor an entity model has to beat, and it is built first for that reason. Bridging
// asks "does the same thing appear in two places" — an entity question — and two chunks sharing
// "Acme" and "8080" have a concrete relationship where two chunks sharing "system" and "the" do
// not. Inverse document frequency is what separates those without a model.
//
// It will not match "the plaintiff" to "Acme Corp"; that is where an entity model earns its place,
// and the comparison is what decides whether it does.
func RareTermRelated(chunks []Chunk) func(a, b Chunk) float64 {
	// Document frequency: how many chunks each term appears in.
	df := map[string]int{}
	terms := make([]map[string]bool, len(chunks))
	byIndex := map[int]int{}

	for i, c := range chunks {
		byIndex[c.Index] = i
		set := map[string]bool{}
		for _, t := range significantTerms(c.Text) {
			set[t] = true
		}
		terms[i] = set
		for t := range set {
			df[t]++
		}
	}
	n := float64(len(chunks))

	return func(a, b Chunk) float64 {
		ia, oka := byIndex[a.Index]
		ib, okb := byIndex[b.Index]
		if !oka || !okb || ia == ib {
			return 0
		}
		var score float64
		for t := range terms[ia] {
			if !terms[ib][t] {
				continue
			}
			// A term in every chunk carries no information about a relationship between two of
			// them; one in exactly these two is the whole signal.
			score += math.Log(n / float64(df[t]))
		}
		return score
	}
}

// significantTerms keeps tokens that could name something: long enough to discriminate, and
// including numbers, because a port or an amount is exactly the kind of shared detail that makes
// two distant passages related.
func significantTerms(s string) []string {
	var out []string
	for _, f := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !isWordRune(r)
	}) {
		if len(f) < 4 || stop[f] {
			continue
		}
		out = append(out, f)
	}
	return out
}

func isWordRune(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')
}
