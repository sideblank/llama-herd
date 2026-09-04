// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

func digests() []Digest {
	return []Digest{
		{Chunk: 0, Data: map[string]any{"system": "A", "port": float64(8080), "tags": []any{"ingest"}}},
		{Chunk: 1, Data: map[string]any{"system": "B", "port": float64(9090), "tags": []any{"report", "ingest"}}},
		{Chunk: 2, Data: map[string]any{"system": "C", "port": float64(7070), "tags": []any{"cache"}}},
	}
}

// The requirement the design turns on. Streams finish in whatever order the scheduler produces, so
// a merge sensitive to arrival order answers the same question differently on different runs — with
// nothing to indicate which answer was which.
func TestMergeIsOrderIndependent(t *testing.T) {
	rules := map[string]Rule{"tags": RuleUnion, "port": RuleAll}

	base, err := Merge(digests(), rules)
	if err != nil {
		t.Fatal(err)
	}
	for trial := 0; trial < 40; trial++ {
		shuffled := digests()
		rand.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })
		got, err := Merge(shuffled, rules)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.Fields, base.Fields) {
			t.Fatalf("arrival order changed the result:\n base %v\n got  %v", base.Fields, got.Fields)
		}
		if !reflect.DeepEqual(got.Conflicts, base.Conflicts) {
			t.Fatalf("arrival order changed the conflicts:\n base %v\n got %v",
				base.Conflicts, got.Conflicts)
		}
	}
}

// Disagreement is information. A merge that silently picks one side produces a confident answer
// that may be the wrong half, and nothing downstream can tell.
func TestDisagreementIsSurfacedNotResolved(t *testing.T) {
	d := []Digest{
		{Chunk: 3, Data: map[string]any{"owner": "Acme"}},
		{Chunk: 19, Data: map[string]any{"owner": "Globex"}},
	}
	got, err := Merge(d, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Conflicts) != 1 {
		t.Fatalf("two chunks disagreed and %d conflicts were reported", len(got.Conflicts))
	}
	c := got.Conflicts[0]
	if c.Field != "owner" || len(c.Values) != 2 {
		t.Fatalf("conflict = %+v", c)
	}
	// A conflict must name its sides, or nothing can resolve it.
	if c.Values[0].Chunk != 3 || c.Values[1].Chunk != 19 {
		t.Errorf("conflict lost provenance: %v", c)
	}
	if !strings.Contains(got.Render(), "disagree") {
		t.Error("the rendered output hides the disagreement")
	}
}

// Chunks agreeing is not a conflict, and must not be reported as one.
func TestAgreementIsNotAConflict(t *testing.T) {
	d := []Digest{
		{Chunk: 0, Data: map[string]any{"owner": "Acme"}},
		{Chunk: 5, Data: map[string]any{"owner": "Acme"}},
	}
	got, _ := Merge(d, nil)
	if len(got.Conflicts) != 0 {
		t.Errorf("identical values reported as a conflict: %v", got.Conflicts)
	}
	if got.Fields["owner"] != "Acme" {
		t.Errorf("owner = %v", got.Fields["owner"])
	}
}

func TestSumIsExact(t *testing.T) {
	d := []Digest{
		{Chunk: 0, Data: map[string]any{"errors": float64(3)}},
		{Chunk: 1, Data: map[string]any{"errors": float64(0)}},
		{Chunk: 2, Data: map[string]any{"errors": float64(11)}},
	}
	got, err := Merge(d, map[string]Rule{"errors": RuleSum})
	if err != nil {
		t.Fatal(err)
	}
	if got.Fields["errors"] != float64(14) {
		t.Errorf("errors = %v, want 14 — counts are the associative case aggregation exists for",
			got.Fields["errors"])
	}
}

// A summed field that is not a number is a broken pipeline, not something to coerce past.
func TestSumRefusesNonNumbers(t *testing.T) {
	d := []Digest{{Chunk: 4, Data: map[string]any{"errors": "several"}}}
	if _, err := Merge(d, map[string]Rule{"errors": RuleSum}); err == nil {
		t.Error("expected an error when a summed field is not numeric")
	}
}

func TestUnionDeduplicatesInDocumentOrder(t *testing.T) {
	got, err := Merge(digests(), map[string]Rule{"tags": RuleUnion})
	if err != nil {
		t.Fatal(err)
	}
	want := []any{"ingest", "report", "cache"}
	if !reflect.DeepEqual(got.Fields["tags"], want) {
		t.Errorf("tags = %v, want %v (first-seen order, which is document order)",
			got.Fields["tags"], want)
	}
}

// Every field must record which chunks contributed, or an answer cannot be attributed or checked.
func TestProvenanceIsKept(t *testing.T) {
	got, _ := Merge(digests(), map[string]Rule{"tags": RuleUnion})
	src := got.Sources["system"]
	if !reflect.DeepEqual(src, []int{0, 1, 2}) {
		t.Errorf("sources for 'system' = %v, want [0 1 2]", src)
	}
}

// A grammar makes the shape valid by construction, so a parse failure means the response was not
// constrained. Dropping it silently would answer from a document with a hole in it.
func TestUnconstrainedResponseIsAnError(t *testing.T) {
	if _, err := ParseDigest(7, "Sure! Here is the JSON you asked for: {\"a\":1}"); err == nil {
		t.Error("prose around the object must be refused, not salvaged")
	}
	if _, err := ParseDigest(7, `{"system":"A","port":8080}`); err != nil {
		t.Errorf("a valid object was refused: %v", err)
	}
}

func TestEmptyMergeIsHarmless(t *testing.T) {
	got, err := Merge(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Chunks != 0 || len(got.Fields) != 0 {
		t.Errorf("empty merge produced %+v", got)
	}
}
