// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func docWithSections(n int) (string, []Chunk) {
	var b strings.Builder
	var chunks []Chunk
	for i := 0; i < n; i++ {
		start := b.Len()
		b.WriteString("Section Heading " + string(rune('A'+i)) + "\n")
		b.WriteString("This section opens by stating its subject clearly. ")
		b.WriteString(strings.Repeat("Filler sentence with no distinguishing content. ", 20))
		b.WriteString("This section closes with its conclusion.\n\n")
		chunks = append(chunks, Chunk{Index: i, Start: start, End: b.Len(), Cut: CutParagraph})
	}
	return b.String(), chunks
}

// The sample must stay within one stream however many chunks there are. A skeleton that does not
// fit cannot be built at all, and it is the pass everything else depends on.
func TestDownsampleStaysBoundedAsChunksGrow(t *testing.T) {
	var last int
	for _, n := range []int{2, 8, 32} {
		src, chunks := docWithSections(n)
		sample := Downsample(chunks, src, 120)
		perChunk := len(sample) / n
		if perChunk > 600 {
			t.Errorf("%d chunks: %d chars per chunk — the sample is not bounded", n, perChunk)
		}
		if len(sample) <= last {
			t.Errorf("%d chunks produced a smaller sample than fewer chunks did", n)
		}
		last = len(sample)
		// The whole document must not be smuggled in wholesale.
		if len(sample) > len(src)/2 {
			t.Errorf("%d chunks: sample is %d of %d chars — that is not a downsample",
				n, len(sample), len(src))
		}
	}
}

// A section states what it is at the start and what it concluded at the end. Both must survive, or
// the skeleton describes only beginnings.
func TestDownsampleKeepsBothEnds(t *testing.T) {
	src, chunks := docWithSections(3)
	sample := Downsample(chunks, src, 120)
	if !strings.Contains(sample, "opens by stating") {
		t.Error("the sample lost section openings")
	}
	if !strings.Contains(sample, "closes with its conclusion") {
		t.Error("the sample lost section endings")
	}
	if !strings.Contains(sample, "Section Heading A") {
		t.Error("the sample lost headings, which are the cheapest structure signal available")
	}
}

// Every section must be represented, or the skeleton describes a document with parts missing and
// nothing downstream knows which.
func TestEverySectionAppears(t *testing.T) {
	src, chunks := docWithSections(6)
	sample := Downsample(chunks, src, 100)
	for i := range chunks {
		if !strings.Contains(sample, "## Section "+string(rune('0'+i))) {
			t.Errorf("section %d is absent from the sample", i)
		}
	}
}

type stubRunner struct {
	got  Request
	resp string
	err  error
}

func (s *stubRunner) Run(_ context.Context, r Request) (string, error) {
	s.got = r
	return s.resp, s.err
}

// The skeleton request must not be attributable to a chunk. A result mis-filed as chunk 0 would
// put an outline where a digest belongs and be merged as content.
func TestSkeletonIsNotAttributedToAChunk(t *testing.T) {
	src, chunks := docWithSections(3)
	st := &stubRunner{resp: "  A supply agreement in five sections.  "}
	got, err := BuildSkeleton(context.Background(), st, chunks, src, 100, 200, fakeCount)
	if err != nil {
		t.Fatal(err)
	}
	if st.got.Chunk != -1 {
		t.Errorf("skeleton request carried chunk %d — it must not pass as a chunk result",
			st.got.Chunk)
	}
	if got != "A supply agreement in five sections." {
		t.Errorf("skeleton = %q (whitespace should be trimmed)", got)
	}
	if !strings.Contains(st.got.Prompt, "identifiers that appear") {
		t.Error("the prompt does not ask for the recurring entities")
	}
	if st.got.MaxTokens > SkeletonCap {
		t.Errorf("skeleton allowed %d tokens against a %d cap — it rides in every context "+
			"downstream and competes with the source it exists to help select",
			st.got.MaxTokens, SkeletonCap)
	}
}

// A failed skeleton must be an error, not an empty string. Silently proceeding without it means
// every chunk is reasoned about in isolation, which is the failure the skeleton exists to prevent.
func TestSkeletonFailureIsAnError(t *testing.T) {
	src, chunks := docWithSections(2)
	st := &stubRunner{err: errors.New("engine unavailable")}
	if _, err := BuildSkeleton(context.Background(), st, chunks, src, 100, 200, fakeCount); err == nil {
		t.Error("a failed skeleton pass returned no error")
	}
}

func TestNoChunksNeedsNoSkeleton(t *testing.T) {
	st := &stubRunner{resp: "should not be called"}
	got, err := BuildSkeleton(context.Background(), st, nil, "", 100, 200, fakeCount)
	if err != nil || got != "" {
		t.Errorf("got %q, %v", got, err)
	}
	if st.got.Prompt != "" {
		t.Error("an empty document still made a request")
	}
}

// A model that ignores its token budget must still not produce an oversized skeleton, and a
// truncated one must not end mid-list — that reads as though the document stops there.
func TestSkeletonIsTruncatedAtASentence(t *testing.T) {
	long := strings.Repeat("This document concerns the platform and its systems. ", 60)
	got := TruncateSkeleton(long, 40, fakeCount)
	if fakeCount(got) > 40 {
		t.Errorf("truncated skeleton is %d tokens against a 40 cap", fakeCount(got))
	}
	if !strings.HasSuffix(got, ".") {
		t.Errorf("truncation did not land on a sentence boundary: %q", got)
	}
}

// Even an unbroken run of text with no sentence to cut at must be bounded rather than passed
// through whole.
func TestSkeletonTruncationHandlesNoSentences(t *testing.T) {
	got := TruncateSkeleton(strings.Repeat("word ", 500), 30, fakeCount)
	if fakeCount(got) > 30 {
		t.Errorf("unbounded: %d tokens", fakeCount(got))
	}
	if got == "" {
		t.Error("truncation produced nothing at all")
	}
}

// A skeleton already within the cap must survive untouched.
func TestShortSkeletonIsUnchanged(t *testing.T) {
	s := "A supply agreement between Acme and Globex, in five sections."
	if got := TruncateSkeleton(s, 120, fakeCount); got != s {
		t.Errorf("a short skeleton was altered: %q", got)
	}
}
