// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// A document with the answer buried in one section, surrounded by irrelevant material — the shape
// virtual context exists to handle.
func buildDoc() (string, *Index) {
	sections := []struct{ topic, body string }{
		{"shipping", "Shipping is dispatched within two business days of an order being placed. "},
		{"returns", "Returns are accepted within thirty days provided the item is unused. "},
		{"termination", "Either party may terminate this agreement with ninety days written notice. " +
			"Termination does not relieve either party of obligations already accrued. "},
		{"payment", "Payment falls due on the first of each month by bank transfer. "},
		{"warranty", "The warranty covers manufacturing defects for a period of one year. "},
	}
	var b strings.Builder
	ix := &Index{Skeleton: "A commercial supply agreement covering shipping, returns, termination, payment and warranty."}
	for i, sec := range sections {
		// Pad so no single span dominates the budget.
		body := strings.Repeat(sec.body, 6)
		start := b.Len()
		b.WriteString(body)
		ix.Spans = append(ix.Spans, Span{
			Start: start, End: b.Len(), Chunk: i, Topic: sec.topic, Weight: 1,
		})
	}
	return b.String(), ix
}

// The point of the whole design: the answer context must contain the SOURCE TEXT relevant to the
// question, verbatim — not a summary of it.
func TestSelectionLoadsRelevantSourceVerbatim(t *testing.T) {
	source, ix := buildDoc()
	sel := Selector{Retriever: LexicalRetriever{}, Count: fakeCount, Budget: 300}

	got, err := sel.Select(context.Background(), "how much notice to terminate the agreement?", ix, source)
	if err != nil {
		t.Fatal(err)
	}
	assembled, err := got.Assemble(ix, source)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(assembled, "ninety days written notice") {
		t.Errorf("the answer context does not contain the passage that answers the question:\n%s",
			assembled)
	}
}

// Selection must fit the budget it was given, or the answer pass is refused at submit time.
func TestSelectionRespectsTheBudget(t *testing.T) {
	source, ix := buildDoc()
	for _, budget := range []int{300, 600, 1200} {
		sel := Selector{Retriever: LexicalRetriever{}, Count: fakeCount, Budget: budget}
		got, err := sel.Select(context.Background(), "termination notice", ix, source)
		if err != nil {
			t.Fatal(err)
		}
		if got.Tokens > budget {
			t.Errorf("budget %d: selected %d tokens", budget, got.Tokens)
		}
	}
}

// Relevance decides what is included; READING ORDER decides how it is assembled. Passages spliced
// out of sequence read as incoherent however well each was chosen.
func TestAssemblyIsInReadingOrder(t *testing.T) {
	source, ix := buildDoc()
	sel := Selector{Retriever: LexicalRetriever{}, Count: fakeCount, Budget: 1200}
	got, err := sel.Select(context.Background(), "termination warranty shipping", ix, source)
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(got.Spans); i++ {
		if got.Spans[i].Start < got.Spans[i-1].Start {
			t.Fatalf("spans out of source order at %d: %d before %d",
				i, got.Spans[i].Start, got.Spans[i-1].Start)
		}
	}
}

// Omissions must be visible. A model handed two passages spliced together with no marker reads
// them as contiguous and reasons about a document that does not exist.
func TestGapsAreMarked(t *testing.T) {
	source, ix := buildDoc()
	sel := Selector{Retriever: LexicalRetriever{}, Count: fakeCount, Budget: 300}
	got, _ := sel.Select(context.Background(), "termination notice", ix, source)
	assembled, _ := got.Assemble(ix, source)

	if len(got.Spans) > 1 && !strings.Contains(assembled, "omitted") {
		t.Error("passages were spliced with no marker for what was left out")
	}
}

// The skeleton rides in every answer context, so nothing is reasoned about in isolation.
func TestSkeletonIsAlwaysPresent(t *testing.T) {
	source, ix := buildDoc()
	sel := Selector{Retriever: LexicalRetriever{}, Count: fakeCount, Budget: 300}
	got, _ := sel.Select(context.Background(), "payment", ix, source)
	assembled, _ := got.Assemble(ix, source)
	if !strings.Contains(assembled, ix.Skeleton) {
		t.Error("the answer context lacks the document overview")
	}
}

// Truncation must be reported. It is the difference between "the document does not say" and
// "we did not look".
func TestTruncationIsReported(t *testing.T) {
	source, ix := buildDoc()
	tight := Selector{Retriever: LexicalRetriever{}, Count: fakeCount, Budget: 300}
	got, _ := tight.Select(context.Background(), "termination", ix, source)
	if !got.Truncated {
		t.Error("a budget too small for every relevant span must report truncation")
	}
}

// An index built from a different input must be refused. Assembling from mismatched offsets
// produces a confident, coherent, wrong answer — the worst outcome this design can have.
func TestMismatchedIndexIsRefused(t *testing.T) {
	source, ix := buildDoc()
	sel := Selector{Retriever: LexicalRetriever{}, Count: fakeCount, Budget: 100}
	if _, err := sel.Select(context.Background(), "termination", ix, source[:len(source)/2]); err == nil {
		t.Error("an index that does not match the source must be refused")
	}
}

// Overlapping spans must not be loaded twice; the duplicate spends budget that would have carried
// something else.
func TestOverlappingSpansAreMerged(t *testing.T) {
	source, ix := buildDoc()
	ix.Spans = append(ix.Spans, Span{Start: 10, End: 200, Chunk: 0, Topic: "overlap", Weight: 1})
	ix.Sort()
	if err := ix.Validate(source); err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(ix.Spans); i++ {
		if ix.Spans[i].Start < ix.Spans[i-1].End {
			t.Errorf("spans %d and %d still overlap after merging", i-1, i)
		}
	}
}

// A budget too small for any span must fail loudly. Returning an empty selection would let the
// answer pass run with no source and answer from the model's own knowledge — confident, grounded
// in nothing, and indistinguishable from a real answer.
func TestNoSpanFittingIsAnError(t *testing.T) {
	source, ix := buildDoc()
	sel := Selector{Retriever: LexicalRetriever{}, Count: fakeCount, Budget: 5}
	if _, err := sel.Select(context.Background(), "termination", ix, source); err == nil {
		t.Error("a budget that fits nothing must be refused, not returned empty")
	}
}

// Budget must never silently swap the evidence for filler. A question about termination answered
// from a passage about shipping is confident, coherent and wrong — the failure this design exists
// to prevent, and it is worse than refusing.
func TestBudgetTooSmallForTheEvidenceIsRefused(t *testing.T) {
	source, ix := buildDoc()
	// 200 fits the shipping passage (111) but not the termination one (224).
	sel := Selector{Retriever: LexicalRetriever{}, Count: fakeCount, Budget: 200}
	_, err := sel.Select(context.Background(), "how much notice to terminate the agreement?", ix, source)
	if err == nil {
		t.Fatal("expected a refusal rather than a selection built from irrelevant text")
	}
	if !strings.Contains(err.Error(), "different question") {
		t.Errorf("the refusal should say why: %v", err)
	}
}

// The failure that made the floor a straw man: a query term appearing in every span counted as
// much as the one identifying the answer. Measured end to end — "port billing node" selected almost
// every span because "node" was in all of them and outvoted "billing".
func TestCommonTermsDoNotDrownTheDiscriminatingOne(t *testing.T) {
	// Every span says "node"; exactly one says "billing".
	var b strings.Builder
	var spans []Span
	for i, role := range []string{"ingest", "reporting", "cache", "search", "billing", "audit"} {
		start := b.Len()
		fmt.Fprintf(&b, "System %d is the %s node of the platform and runs in the primary region. ",
			i, role)
		spans = append(spans, Span{Start: start, End: b.Len(), Chunk: i, Weight: 1})
	}
	source := b.String()
	ix := &Index{Spans: spans}

	ranked, err := LexicalRetriever{}.Rank(context.Background(), "which node handles billing", source, ix.Spans)
	if err != nil {
		t.Fatal(err)
	}
	top, err := ranked[0].Text(source)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(top, "billing") {
		t.Errorf("top-ranked span is %q — the discriminating term lost to the common one", top)
	}
}

// With ranking that works, a tight budget must still find the answer — which is the whole point of
// selecting rather than loading everything.
func TestTightBudgetStillSelectsTheRightSpan(t *testing.T) {
	var b strings.Builder
	var spans []Span
	for i, role := range []string{"ingest", "reporting", "cache", "search", "billing", "audit"} {
		start := b.Len()
		fmt.Fprintf(&b, "System %d is the %s node and uses port %d000 for inbound traffic. ",
			i, role, i+4)
		spans = append(spans, Span{Start: start, End: b.Len(), Chunk: i, Weight: 1})
	}
	source := b.String()
	ix := &Index{Spans: spans}

	// Room for roughly one span.
	sel, err := (Selector{Retriever: LexicalRetriever{}, Count: fakeCount, Budget: 24}).
		Select(context.Background(), "which node handles billing", ix, source)
	if err != nil {
		t.Fatal(err)
	}
	var joined strings.Builder
	for _, sp := range sel.Spans {
		txt, _ := sp.Text(source)
		joined.WriteString(txt)
	}
	if !strings.Contains(joined.String(), "billing") {
		t.Errorf("a tight budget selected %q — ranking did not put the answer first",
			joined.String())
	}
	if !sel.Truncated {
		t.Error("spans were left out but truncation was not reported")
	}
}
