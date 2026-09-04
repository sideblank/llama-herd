// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import "strings"

// Segment divides a chunk into candidate spans at natural boundaries, with exact offsets.
//
// Offsets are computed here and never asked of a model. A language model cannot count characters,
// and an offset that is plausibly wrong is the worst kind: text is assembled from the wrong region
// and the answer is confident, coherent and about something else. Locating spans is arithmetic;
// only labelling them needs judgement.
//
// Segments are paragraphs, or sentences where a paragraph exceeds maxTokens — large enough to carry
// a complete thought, small enough that selecting one does not spend the answer budget on
// surrounding material that was not asked for.
func Segment(chunk Chunk, source string, maxTokens int, count CountTokens) []Span {
	if count == nil || chunk.End <= chunk.Start || chunk.End > len(source) {
		return nil
	}
	if maxTokens < 1 {
		maxTokens = 256
	}

	var out []Span
	body := source[chunk.Start:chunk.End]

	for _, para := range splitKeepingOffsets(body, "\n\n") {
		text := source[chunk.Start+para.start : chunk.Start+para.end]
		if strings.TrimSpace(text) == "" {
			continue
		}
		if count(text) <= maxTokens {
			out = append(out, Span{
				Start: chunk.Start + para.start,
				End:   chunk.Start + para.end,
				Chunk: chunk.Index,
			})
			continue
		}
		// Too large to select usefully as one unit: fall back to sentences, which is the
		// smallest division that still carries a complete assertion.
		for _, sent := range splitSentences(text) {
			if strings.TrimSpace(source[chunk.Start+para.start+sent.start:chunk.Start+para.start+sent.end]) == "" {
				continue
			}
			out = append(out, Span{
				Start: chunk.Start + para.start + sent.start,
				End:   chunk.Start + para.start + sent.end,
				Chunk: chunk.Index,
			})
		}
	}
	return out
}

type piece struct{ start, end int }

// splitKeepingOffsets divides on sep while tracking positions, because the offsets are the point:
// strings.Split discards exactly the information the index needs.
func splitKeepingOffsets(s, sep string) []piece {
	var out []piece
	start := 0
	for {
		i := strings.Index(s[start:], sep)
		if i < 0 {
			if start < len(s) {
				out = append(out, piece{start, len(s)})
			}
			return out
		}
		end := start + i
		if end > start {
			out = append(out, piece{start, end})
		}
		start = end + len(sep)
		if start >= len(s) {
			return out
		}
	}
}

// splitSentences divides on terminators, keeping the terminator with its sentence — a span ending
// before its full stop reads as truncated to whatever consumes it.
func splitSentences(s string) []piece {
	var out []piece
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] != '.' && s[i] != '!' && s[i] != '?' {
			continue
		}
		// A terminator is only a boundary when whitespace or the end follows it, so decimals
		// and abbreviations do not split a sentence in half.
		if i+1 < len(s) && s[i+1] != ' ' && s[i+1] != '\n' && s[i+1] != '\t' {
			continue
		}
		end := i + 1
		if end > start {
			out = append(out, piece{start, end})
		}
		start = end
		for start < len(s) && (s[start] == ' ' || s[start] == '\n' || s[start] == '\t') {
			start++
		}
		i = start - 1
	}
	if start < len(s) {
		out = append(out, piece{start, len(s)})
	}
	return out
}

// SegmentAll indexes every chunk, producing spans in document order.
func SegmentAll(chunks []Chunk, source string, maxTokens int, count CountTokens) *Index {
	ix := &Index{}
	for _, c := range chunks {
		ix.Spans = append(ix.Spans, Segment(c, source, maxTokens, count)...)
	}
	ix.Sort()
	return ix
}
