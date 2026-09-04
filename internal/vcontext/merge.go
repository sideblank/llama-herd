// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Digest is one chunk's structured output, produced under a grammar so its shape is guaranteed.
type Digest struct {
	Chunk int
	Data  map[string]any
}

// ParseDigest reads one chunk's grammar-constrained response.
//
// A grammar makes the shape valid by construction, so a failure here means the response did not
// come from a constrained stream — worth an error rather than a skip, because silently dropping a
// chunk answers the question from a document with a hole in it.
func ParseDigest(chunk int, raw string) (Digest, error) {
	var m map[string]any
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		return Digest{}, fmt.Errorf("vcontext: chunk %d did not return an object (%w) — a "+
			"grammar-constrained stream cannot do this, so this response was unconstrained",
			chunk, err)
	}
	return Digest{Chunk: chunk, Data: m}, nil
}

// Rule says how one field combines across chunks.
type Rule int

const (
	// RuleCollect gathers distinct values and reports a conflict when chunks disagree.
	//
	// The default, deliberately. Disagreement between chunks is information — two parts of a
	// document saying different things is exactly what a reader needs to know — and a merge that
	// silently picks one produces a confident answer that may be the wrong half.
	RuleCollect Rule = iota
	// RuleSum adds numbers. For counts, which are the associative case aggregation exists for.
	RuleSum
	// RuleUnion concatenates arrays and removes duplicates.
	RuleUnion
	// RuleAll keeps every value with its chunk, never conflicting. For per-chunk observations
	// that are all true at once.
	RuleAll
)

// Value is one contribution, with the chunk it came from.
//
// Provenance is not decoration: an answer that cannot say where a fact came from cannot be checked,
// and a conflict that cannot name its sides cannot be resolved by anything downstream.
type Value struct {
	Chunk int
	V     any
}

// Conflict is a field where chunks disagreed under RuleCollect.
type Conflict struct {
	Field  string
	Values []Value
}

func (c Conflict) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: ", c.Field)
	for i, v := range c.Values {
		if i > 0 {
			b.WriteString(" vs ")
		}
		fmt.Fprintf(&b, "%v (chunk %d)", v.V, v.Chunk)
	}
	return b.String()
}

// Merged is the combination of every chunk's digest.
type Merged struct {
	// Fields holds the combined value per key.
	Fields map[string]any
	// Sources records which chunks contributed to each key.
	Sources map[string][]int
	// Conflicts are disagreements, surfaced rather than resolved. Empty means the chunks agreed.
	Conflicts []Conflict
	// Chunks is how many digests went in.
	Chunks int
}

// Merge combines digests deterministically.
//
// Order independence is a correctness requirement, not a nicety: streams finish in whatever order
// the scheduler produces, so a merge sensitive to arrival order would answer the same question
// differently on different runs, with nothing to indicate which answer was which. Every rule here
// is commutative, and the digests are sorted by chunk before combining so any ordering inside a
// result reflects the document rather than the race.
func Merge(digests []Digest, rules map[string]Rule) (Merged, error) {
	out := Merged{
		Fields:  map[string]any{},
		Sources: map[string][]int{},
		Chunks:  len(digests),
	}
	if len(digests) == 0 {
		return out, nil
	}

	// Sort by chunk so anything order-sensitive downstream sees document order, never
	// completion order.
	ds := append([]Digest(nil), digests...)
	sort.SliceStable(ds, func(i, j int) bool { return ds[i].Chunk < ds[j].Chunk })

	gathered := map[string][]Value{}
	var keys []string
	for _, d := range ds {
		for k, v := range d.Data {
			if _, seen := gathered[k]; !seen {
				keys = append(keys, k)
			}
			gathered[k] = append(gathered[k], Value{Chunk: d.Chunk, V: v})
			out.Sources[k] = append(out.Sources[k], d.Chunk)
		}
	}
	sort.Strings(keys) // stable field order regardless of map iteration

	for _, k := range keys {
		vals := gathered[k]
		switch rules[k] {
		case RuleSum:
			total, err := sumValues(k, vals)
			if err != nil {
				return Merged{}, err
			}
			out.Fields[k] = total
		case RuleUnion:
			out.Fields[k] = unionValues(vals)
		case RuleAll:
			all := make([]any, 0, len(vals))
			for _, v := range vals {
				all = append(all, v.V)
			}
			out.Fields[k] = all
		default: // RuleCollect
			distinct := distinctValues(vals)
			if len(distinct) == 1 {
				out.Fields[k] = distinct[0].V
				continue
			}
			out.Fields[k] = firstOf(distinct)
			out.Conflicts = append(out.Conflicts, Conflict{Field: k, Values: distinct})
		}
	}
	sort.SliceStable(out.Conflicts, func(i, j int) bool {
		return out.Conflicts[i].Field < out.Conflicts[j].Field
	})
	return out, nil
}

func sumValues(field string, vals []Value) (float64, error) {
	var total float64
	for _, v := range vals {
		n, ok := v.V.(float64) // encoding/json numbers
		if !ok {
			return 0, fmt.Errorf("vcontext: field %q is summed but chunk %d returned %T — a "+
				"grammar should have made that impossible", field, v.Chunk, v.V)
		}
		total += n
	}
	return total, nil
}

// unionValues flattens arrays across chunks and removes duplicates, keeping first-seen order,
// which is document order because the digests were sorted.
func unionValues(vals []Value) []any {
	var out []any
	seen := map[string]bool{}
	for _, v := range vals {
		items, ok := v.V.([]any)
		if !ok {
			items = []any{v.V}
		}
		for _, it := range items {
			k := fmt.Sprintf("%v", it)
			if seen[k] {
				continue
			}
			seen[k] = true
			out = append(out, it)
		}
	}
	return out
}

// distinctValues keeps one Value per distinct content, in document order.
func distinctValues(vals []Value) []Value {
	var out []Value
	seen := map[string]bool{}
	for _, v := range vals {
		k := fmt.Sprintf("%v", v.V)
		if seen[k] {
			continue
		}
		seen[k] = true
		out = append(out, v)
	}
	return out
}

func firstOf(vals []Value) any {
	if len(vals) == 0 {
		return nil
	}
	return vals[0].V
}

// Render turns a merge into text for the answer pass, conflicts included.
//
// Conflicts are rendered, never dropped. A document that says two different things is a fact about
// the document, and the answer should be able to say so rather than presenting one side as settled.
func (m Merged) Render() string {
	var b strings.Builder
	fmt.Fprintf(&b, "Combined from %d sections.\n\n", m.Chunks)

	keys := make([]string, 0, len(m.Fields))
	for k := range m.Fields {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Fprintf(&b, "%s: %v\n", k, m.Fields[k])
	}
	if len(m.Conflicts) > 0 {
		b.WriteString("\nSections disagree on:\n")
		for _, c := range m.Conflicts {
			fmt.Fprintf(&b, "  %s\n", c)
		}
	}
	return b.String()
}
