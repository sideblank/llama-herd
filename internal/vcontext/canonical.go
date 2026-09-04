// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"regexp"
	"strings"
)

// Canonicalisation trims text before it is tokenised, so prefill pays for fewer tokens.
//
// Worth doing because prefill dominates: at ~3,300 tok/s on a 3090, 256k of input is ~78 seconds
// and every token removed is ~0.3 ms. A 12% reduction is ~9 seconds off the critical path, and it
// costs microseconds of regex.
//
// ⚠️ It must run ONCE, BEFORE chunking, and its output treated as the source from then on. The span
// index is built on exact byte offsets, so canonicalising afterwards would move every span off the
// text it names — assembling the answer from the neighbouring region, confidently and wrongly.
//
// ⚠️ And it is not "remove whitespace". BPE vocabularies carry single tokens for a leading space
// with a word, and for "\n\n". Stripping those breaks the merge and the tokenizer falls back to
// fragments or raw bytes, INCREASING the count. Every pass below is therefore a candidate to be
// measured, not a rule to be trusted — see CanonPasses and the `canon` command.

// CanonPass is one named transformation, kept separate so its effect can be measured alone.
//
// Separate on purpose: the only way to know a pass helps is to tokenise with and without it, and a
// monolithic function cannot be bisected when the total comes out worse.
type CanonPass struct {
	Name  string
	Why   string
	Apply func(string) string
}

var (
	reTrailingSpaces = regexp.MustCompile(`[ \t]+\n`)
	reInlineSpaces   = regexp.MustCompile(`[ \t]{2,}`)
	reExcessNewlines = regexp.MustCompile(`\n{3,}`)
	reSoftBreak      = regexp.MustCompile(`([^\n\s])\n([^\n\s\-\*\d>#|])`)
	reFence          = regexp.MustCompile("(?s)```.*?```|(?s)~~~.*?~~~")
)

// CanonPasses are the transformations, in the order they should run.
func CanonPasses() []CanonPass {
	return []CanonPass{
		{
			Name: "trailing-space",
			Why:  "spaces before a newline are always wasted; nothing merges across them",
			Apply: func(s string) string {
				return outsideFences(s, func(t string) string {
					return reTrailingSpaces.ReplaceAllString(t, "\n")
				})
			},
		},
		{
			Name: "collapse-inline-space",
			Why: "runs of spaces tokenise as repeated 2-space fragments; one space keeps the " +
				"leading-space merge that BPE has a single token for",
			Apply: func(s string) string {
				// Two or more only: a single space is load-bearing and must survive.
				// Outside fences only: this pass destroys code indentation.
				return outsideFences(s, func(t string) string {
					return reInlineSpaces.ReplaceAllString(t, " ")
				})
			},
		},
		{
			Name: "join-soft-breaks",
			Why: "hard-wrapped prose costs a newline token per line and splits words that would " +
				"otherwise merge; skipped inside code fences, where a join changes meaning",
			Apply: joinSoftBreaks,
		},
		{
			Name: "cap-newlines",
			Why:  `"\n\n" is one token; four newlines are two, for no added meaning`,
			Apply: func(s string) string {
				return outsideFences(s, func(t string) string {
					return reExcessNewlines.ReplaceAllString(t, "\n\n")
				})
			},
		},
	}
}

// outsideFences applies f to prose only, passing fenced code through verbatim.
//
// Every pass needs this, not just line-joining — which is the lesson from getting it wrong. Space
// collapsing looks harmless until it turns four spaces of indentation into one, and in a
// whitespace-significant language that is not cosmetic, it is a syntax change. A document carrying
// a configuration file, a stack trace or a code sample would be silently corrupted, and the answer
// would then be about text that does not exist.
//
// Canonicalisation is for prose. Code is left exactly as it was found.
func outsideFences(s string, f func(string) string) string {
	var b strings.Builder
	last := 0
	for _, loc := range reFence.FindAllStringIndex(s, -1) {
		b.WriteString(f(s[last:loc[0]]))
		b.WriteString(s[loc[0]:loc[1]]) // fence verbatim
		last = loc[1]
	}
	b.WriteString(f(s[last:]))
	return b.String()
}

func joinSoftBreaks(s string) string {
	return outsideFences(s, func(t string) string {
		return reSoftBreak.ReplaceAllString(t, "$1 $2")
	})
}

// Canonicalise applies every pass, or only those named.
func Canonicalise(s string, only ...string) string {
	keep := map[string]bool{}
	for _, n := range only {
		keep[n] = true
	}
	for _, p := range CanonPasses() {
		if len(only) > 0 && !keep[p.Name] {
			continue
		}
		s = p.Apply(s)
	}
	return strings.TrimSpace(s)
}

// CanonResult is one pass measured against a real tokenizer.
type CanonResult struct {
	Name        string
	Why         string
	BeforeChars int
	AfterChars  int
	BeforeToks  int
	AfterToks   int
}

// Saved is tokens removed. Negative means the pass broke a merge and cost tokens.
func (c CanonResult) Saved() int { return c.BeforeToks - c.AfterToks }

// Helped reports whether the pass earned its place on this text.
func (c CanonResult) Helped() bool { return c.Saved() > 0 }

// MeasureCanon applies each pass in turn, tokenising before and after.
//
// This is the validation rule made structural rather than left as advice: a pass that increases the
// count has broken a BPE merge and should be dropped for that corpus. Estimates do not settle it,
// because the answer depends on the tokenizer and the text — the same regex that saves 5% on prose
// can cost tokens on indented code.
func MeasureCanon(s string, count CountTokens) []CanonResult {
	var out []CanonResult
	for _, p := range CanonPasses() {
		before := s
		after := p.Apply(s)
		out = append(out, CanonResult{
			Name:        p.Name,
			Why:         p.Why,
			BeforeChars: len(before),
			AfterChars:  len(after),
			BeforeToks:  count(before),
			AfterToks:   count(after),
		})
		s = after
	}
	return out
}
