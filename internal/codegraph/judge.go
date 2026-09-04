// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package codegraph

import (
	"fmt"
	"sort"
	"strings"
)

// Role is what a chunk claimed about a symbol.
type Role string

const (
	Defines Role = "defines"
	Uses    Role = "uses"
)

// Assertion is one chunk's structured claim about one symbol.
//
// Emitted by the local judging pass under a grammar. Deliberately small: the aggregation is a Go
// map lookup, and everything that makes it slower makes the 48-way fan-out worse.
type Assertion struct {
	Chunk  int    `json:"chunk"`
	Symbol string `json:"symbol"`
	Role   Role   `json:"role"`
	Arity  int    `json:"arity"`
	Sig    string `json:"sig"`
	Detail string `json:"detail"`
}

// ConflictKind classifies a cross-chunk contradiction.
type ConflictKind string

const (
	ArityMismatch     ConflictKind = "arity"
	SignatureMismatch ConflictKind = "signature"
	Undefined         ConflictKind = "undefined"
	Redefined         ConflictKind = "redefined"
)

// Conflict is a contradiction between chunks that no single chunk could have seen.
type Conflict struct {
	Kind        ConflictKind
	Symbol      string
	Definitions []Assertion
	Uses        []Assertion
}

func (c Conflict) String() string {
	chunks := func(as []Assertion) string {
		var s []string
		for _, a := range as {
			s = append(s, fmt.Sprintf("chunk %d", a.Chunk))
		}
		return strings.Join(s, ", ")
	}
	switch c.Kind {
	case ArityMismatch:
		return fmt.Sprintf("%s: defined taking %d args (%s), called with %d (%s)",
			c.Symbol, c.Definitions[0].Arity, chunks(c.Definitions),
			c.Uses[0].Arity, chunks(c.Uses))
	case SignatureMismatch:
		return fmt.Sprintf("%s: defined as %q (%s), used as %q (%s)",
			c.Symbol, c.Definitions[0].Sig, chunks(c.Definitions),
			c.Uses[0].Sig, chunks(c.Uses))
	case Undefined:
		return fmt.Sprintf("%s: used in %s, defined nowhere in the examined chunks",
			c.Symbol, chunks(c.Uses))
	default:
		return fmt.Sprintf("%s: defined in %s with differing signatures", c.Symbol, chunks(c.Definitions))
	}
}

// Coverage is what the judging pass actually looked at.
//
// Reported with every verdict and never separable from it. A cross-check can only contradict claims
// that were made: a symbol nobody asserted about, or a chunk that returned nothing, is not evidence
// of correctness — it is an unexamined region. Without these numbers "no conflicts" and "nothing was
// checked" are the same output.
type Coverage struct {
	Chunks          int
	ChunksReporting int
	Symbols         int
	Definitions     int
	Uses            int
	// Silent lists chunks that produced no assertions at all. Their content was never judged,
	// whatever the verdict says about everything else.
	Silent []int
	// Unasserted lists symbols from the global table that no chunk mentioned. Each one is a
	// declaration nothing was checked against.
	Unasserted []string
}

// Complete reports whether every chunk answered and every known symbol was mentioned.
func (c Coverage) Complete() bool { return len(c.Silent) == 0 && len(c.Unasserted) == 0 }

// Verdict is the outcome of the cross-chunk check.
//
// There is deliberately no Correct() method and no boolean named "correct". The cross-check finds
// contradictions between chunks; it does not and cannot establish that code is right. It is blind
// to a logic error wholly inside one chunk that the local pass missed, to a wrong-but-consistent
// contract agreed on by both sides, and to anything in a region that returned no assertions.
// "Zero conflicts across 48 chunks" is a statement about the checks that ran, and naming it
// CORRECT would convert an absence of evidence into a claim.
type Verdict struct {
	Conflicts []Conflict
	Coverage  Coverage
}

// NoConflictsFound reports that the cross-check found no contradiction. Read the name literally: it
// is not a correctness result, and it means little without Coverage beside it.
func (v Verdict) NoConflictsFound() bool { return len(v.Conflicts) == 0 }

// Summary always renders coverage next to the result, so the two cannot be quoted apart.
func (v Verdict) Summary() string {
	var b strings.Builder
	c := v.Coverage
	if len(v.Conflicts) == 0 {
		fmt.Fprintf(&b, "no cross-chunk conflicts found")
	} else {
		fmt.Fprintf(&b, "%d cross-chunk conflict(s)", len(v.Conflicts))
	}
	fmt.Fprintf(&b, " — checked %d symbols (%d definitions, %d uses) across %d/%d chunks",
		c.Symbols, c.Definitions, c.Uses, c.ChunksReporting, c.Chunks)
	if len(c.Silent) > 0 {
		fmt.Fprintf(&b, "; %d chunk(s) returned nothing and were NOT judged: %v", len(c.Silent), c.Silent)
	}
	if n := len(c.Unasserted); n > 0 {
		show := c.Unasserted
		if n > 5 {
			show = show[:5]
		}
		fmt.Fprintf(&b, "; %d declared symbol(s) unmentioned by any chunk (%s%s)",
			n, strings.Join(show, ", "), map[bool]string{true: ", …"}[n > 5])
	}
	if !c.Complete() {
		b.WriteString(". This is a partial result.")
	}
	return b.String()
}

// CrossCheck aggregates per-chunk assertions and reports contradictions.
//
// chunks is the total dispatched, so silence can be distinguished from agreement. known is the
// global symbol table, used to report what nothing asserted about; pass nil to skip that.
func CrossCheck(chunks int, assertions []Assertion, known []string) Verdict {
	defs := map[string][]Assertion{}
	uses := map[string][]Assertion{}
	reporting := map[int]bool{}
	mentioned := map[string]bool{}

	for _, a := range assertions {
		if a.Symbol == "" {
			continue
		}
		reporting[a.Chunk] = true
		key := norm(a.Symbol)
		mentioned[key] = true
		if a.Role == Defines {
			defs[key] = append(defs[key], a)
		} else {
			uses[key] = append(uses[key], a)
		}
	}

	var symbols []string
	for s := range defs {
		symbols = append(symbols, s)
	}
	for s := range uses {
		if _, ok := defs[s]; !ok {
			symbols = append(symbols, s)
		}
	}
	sort.Strings(symbols)

	var conflicts []Conflict
	for _, s := range symbols {
		d, u := defs[s], uses[s]
		sort.Slice(d, func(i, j int) bool { return d[i].Chunk < d[j].Chunk })
		sort.Slice(u, func(i, j int) bool { return u[i].Chunk < u[j].Chunk })

		if len(d) == 0 {
			conflicts = append(conflicts, Conflict{Kind: Undefined, Symbol: s, Uses: u})
			continue
		}
		// Two chunks defining the same symbol differently: in most languages a redeclaration,
		// and the local passes cannot see each other to notice.
		for i := 1; i < len(d); i++ {
			if !sameClaim(d[0], d[i]) {
				conflicts = append(conflicts, Conflict{
					Kind: Redefined, Symbol: s, Definitions: []Assertion{d[0], d[i]},
				})
				break
			}
		}
		for _, use := range u {
			switch {
			case use.Arity > 0 && d[0].Arity > 0 && use.Arity != d[0].Arity:
				conflicts = append(conflicts, Conflict{
					Kind: ArityMismatch, Symbol: s,
					Definitions: []Assertion{d[0]}, Uses: []Assertion{use},
				})
			case use.Sig != "" && d[0].Sig != "" && squash(use.Sig) != squash(d[0].Sig):
				conflicts = append(conflicts, Conflict{
					Kind: SignatureMismatch, Symbol: s,
					Definitions: []Assertion{d[0]}, Uses: []Assertion{use},
				})
			}
		}
	}

	cov := Coverage{
		Chunks:          chunks,
		ChunksReporting: len(reporting),
		Symbols:         len(symbols),
		Definitions:     len(defs),
		Uses:            len(uses),
	}
	for i := 0; i < chunks; i++ {
		if !reporting[i] {
			cov.Silent = append(cov.Silent, i)
		}
	}
	for _, k := range known {
		if !mentioned[norm(k)] {
			cov.Unasserted = append(cov.Unasserted, k)
		}
	}
	sort.Strings(cov.Unasserted)
	sort.Slice(conflicts, func(i, j int) bool { return conflicts[i].Symbol < conflicts[j].Symbol })
	return Verdict{Conflicts: conflicts, Coverage: cov}
}

// sameClaim compares two definitions of one symbol, ignoring formatting only.
func sameClaim(a, b Assertion) bool {
	if a.Arity != b.Arity {
		return false
	}
	return squash(a.Sig) == squash(b.Sig)
}

// AssertionGrammar constrains the local judging pass so aggregation never has to parse prose.
const AssertionGrammar = `root  ::= "{" ws "\"assertions\":" ws "[" ws (a (ws "," ws a)*)? ws "]" ws "}"
a     ::= "{" ws "\"symbol\":" ws str "," ws "\"role\":" ws role "," ws "\"arity\":" ws int "," ws "\"sig\":" ws str ws "}"
role  ::= "\"defines\"" | "\"uses\""
str   ::= "\"" [^"\\]* "\""
int   ::= "-"? [0-9]+
ws    ::= [ \t\n]*`
