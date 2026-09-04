// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"fmt"
	"strings"
)

// Downsample builds a compressed view of the whole input for the skeleton pass.
//
// The skeleton must describe the entire document, and the entire document does not fit — that is
// the problem this layer exists for. So it is built from a sample: the opening and closing of each
// chunk, which is where a section states what it is and what it concluded, plus any line that looks
// like a heading.
//
// perChunk bounds what each chunk contributes, so the sample stays within one stream regardless of
// how many chunks there are. A skeleton that does not fit cannot be built at all.
func Downsample(chunks []Chunk, source string, perChunk int) string {
	if perChunk < 40 {
		perChunk = 40
	}
	var b strings.Builder
	for _, c := range chunks {
		if c.End > len(source) || c.Start >= c.End {
			continue
		}
		body := source[c.Start:c.End]
		fmt.Fprintf(&b, "## Section %d\n", c.Index)

		if h := headings(body); len(h) > 0 {
			for _, line := range h {
				fmt.Fprintf(&b, "%s\n", line)
			}
		}
		// Head and tail: a section usually says what it is at the start and what it concluded at
		// the end. The middle is what selection is for.
		head, tail := ends(body, perChunk)
		b.WriteString(head)
		if tail != "" {
			b.WriteString("\n[...]\n")
			b.WriteString(tail)
		}
		b.WriteString("\n\n")
	}
	return b.String()
}

// ends returns the first and last n characters, cut at word boundaries so the sample does not end
// mid-token and invite the model to complete it.
func ends(s string, n int) (head, tail string) {
	if len(s) <= 2*n {
		return s, ""
	}
	head = s[:n]
	if i := strings.LastIndexAny(head, " \n\t"); i > n/2 {
		head = head[:i]
	}
	tail = s[len(s)-n:]
	if i := strings.IndexAny(tail, " \n\t"); i >= 0 && i < n/2 {
		tail = tail[i+1:]
	}
	return head, tail
}

// headings picks out lines that look like structure rather than prose: short, and not ending in a
// sentence terminator.
func headings(s string) []string {
	var out []string
	for _, line := range strings.Split(s, "\n") {
		t := strings.TrimSpace(line)
		if t == "" || len(t) > 80 {
			continue
		}
		if strings.HasPrefix(t, "#") || strings.HasPrefix(t, "##") {
			out = append(out, t)
			continue
		}
		if !strings.HasSuffix(t, ".") && !strings.HasSuffix(t, ",") && len(strings.Fields(t)) <= 8 {
			out = append(out, t)
		}
		if len(out) >= 4 {
			break
		}
	}
	return out
}

// SkeletonPrompt is what the skeleton pass asks for.
//
// It asks for structure and named things, not a summary. A summary of 256k is either useless or
// enormous; what every downstream context needs is a map — what this is, how it is organised, and
// what recurs — so a chunk can resolve "the plaintiff" and the answer pass knows the shape of the
// parts it did not load.
const SkeletonPrompt = `The following is a compressed sample of a longer document: the opening and
closing of each section, in order.

In no more than 80 words, state:
- what kind of document this is
- the people, organisations, systems or identifiers that appear in it

Be terse. Do not summarise the content, do not use headings, and do not repeat a section's details.`

// SkeletonCap bounds the skeleton in tokens, hard.
//
// It rides in every context downstream, so it competes for the same budget as the source it exists
// to help select from — measured: an unbounded skeleton took 300 of a 900-token stream, a third of
// the space, to describe six sections it then got wrong. A map that crowds out the territory is
// worse than no map.
const SkeletonCap = 120

// BuildSkeleton runs the skeleton pass and returns the outline.
//
// One request, on the downsampled view. It is the cheapest pass in the pipeline and the one every
// other pass depends on, because it is what stops each chunk being reasoned about in isolation.
func BuildSkeleton(ctx context.Context, r Runner, chunks []Chunk, source string,
	perChunk, maxTokens int, count CountTokens) (string, error) {

	if len(chunks) == 0 {
		return "", nil
	}
	sample := Downsample(chunks, source, perChunk)
	if maxTokens <= 0 || maxTokens > SkeletonCap {
		maxTokens = SkeletonCap
	}
	text, err := r.Run(ctx, Request{
		Chunk:     -1, // not a chunk: -1 so a mis-attributed result cannot pass as one
		Prompt:    SkeletonPrompt + "\n\n---\n\n" + sample,
		MaxTokens: maxTokens,
	})
	if err != nil {
		return "", fmt.Errorf("vcontext: skeleton pass failed: %w", err)
	}
	return TruncateSkeleton(strings.TrimSpace(text), maxTokens, count), nil
}

// TruncateSkeleton enforces the cap even when the model ignored it.
//
// max_tokens bounds generation, but a model that writes headings and lists still fills it, and a
// truncated skeleton ending mid-list is worse than a short one — it reads as though the document
// stops there. Cut at the last sentence that fits.
func TruncateSkeleton(s string, maxTokens int, count CountTokens) string {
	if count == nil || count(s) <= maxTokens {
		return s
	}
	// Back off a sentence at a time rather than cutting mid-clause.
	sentences := splitSentences(s)
	var keep int
	for _, p := range sentences {
		if count(s[:p.end]) > maxTokens {
			break
		}
		keep = p.end
	}
	if keep == 0 {
		// Not even one sentence fits: take what does, at a word boundary.
		cut := len(s)
		for cut > 0 && count(s[:cut]) > maxTokens {
			cut = cut * 3 / 4
		}
		if i := strings.LastIndexAny(s[:cut], " \n"); i > 0 {
			cut = i
		}
		return strings.TrimSpace(s[:cut])
	}
	return strings.TrimSpace(s[:keep])
}
