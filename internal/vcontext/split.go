// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"fmt"
	"strings"
)

// CountTokens measures a string the way the target model will. Supplied by the caller because
// this package must not depend on a tokenizer, and because an estimate here produces chunks that
// overflow their stream at submit time — far from the cause.
type CountTokens func(string) int

// CutQuality records how cleanly a chunk's trailing boundary fell.
//
// It is the priority signal for bridging. A chunk ending at a paragraph break probably severed
// nothing; one ending mid-sentence certainly did. Spare streams are finite, so they go to the
// boundaries most likely to have broken something.
type CutQuality int

const (
	// CutEnd is the end of the input — nothing was severed.
	CutEnd CutQuality = iota
	// CutDeclaration fell on a structural boundary supplied by a parser: the end of a
	// top-level function, type or class.
	//
	// The best real cut there is, and the only one that can promise nothing was severed. A
	// paragraph break is a good guess about prose; a declaration boundary is a fact about
	// code. Cutting a function in half blinds the local pass to its own body, and the pass
	// still answers — about a fragment, confidently.
	CutDeclaration
	// CutParagraph fell on a blank line. Least likely to have broken a relationship.
	CutParagraph
	// CutSentence fell on a sentence terminator.
	CutSentence
	// CutWord fell on whitespace: mid-sentence, and something was almost certainly severed.
	CutWord
	// CutHard fell mid-word, because no boundary existed within the window.
	CutHard
)

func (c CutQuality) String() string {
	switch c {
	case CutDeclaration:
		return "declaration"
	case CutEnd:
		return "end-of-input"
	case CutParagraph:
		return "paragraph"
	case CutSentence:
		return "sentence"
	case CutWord:
		return "word"
	default:
		return "hard"
	}
}

// Chunk is one piece of a split input.
type Chunk struct {
	Index  int
	Text   string
	Tokens int
	// Start and End are byte offsets into the original input, so a chunk can be related back
	// to the source and bridged against its neighbours.
	Start, End int
	// Cut is how cleanly this chunk's trailing edge fell. See CutQuality.
	Cut CutQuality
}

// Split divides text into the shape a Plan calls for, cutting at boundaries that preserve meaning
// where it can.
//
// The cut hierarchy is paragraph, then sentence, then whitespace, then hard. Each level down is a
// worse cut, so the search takes the best boundary within a window near the target rather than the
// nearest boundary of any kind — a paragraph break slightly early beats a mid-sentence break
// exactly on target.
//
// Chunks never exceed the policy's usable size. A chunk that does not fit its stream is refused
// at submit time by the engine, which surfaces as a failed request rather than as a splitting bug.
// Split cuts text into chunks on prose boundaries.
func Split(text string, plan Plan, count CountTokens) ([]Chunk, error) {
	return SplitAt(text, plan, count, nil)
}

// SplitAt cuts text, preferring the structural boundaries in bounds.
//
// bounds holds absolute byte offsets into text where a top-level declaration ends, ascending. They
// are preferences, not constraints: a boundary is used when one falls inside the backup window, and
// the prose ladder is the fallback everywhere else. That matters because a real file has long
// stretches with no declaration boundary at all — a 900-line function has exactly two — and
// treating them as constraints would produce chunks that do not fit a stream.
//
// Supplying boundaries for a language whose parser failed is safe: an empty slice is the prose
// behaviour, which is what a partially-parsed file should fall back to.
func SplitAt(text string, plan Plan, count CountTokens, bounds []int) ([]Chunk, error) {
	if count == nil {
		return nil, fmt.Errorf("vcontext: a token counter is required — estimating produces " +
			"chunks that overflow their stream")
	}
	if plan.Refused {
		return nil, fmt.Errorf("vcontext: %s", plan.Reason)
	}
	if plan.Chunks <= 1 {
		return []Chunk{{Index: 0, Text: text, Tokens: count(text),
			Start: 0, End: len(text), Cut: CutEnd}}, nil
	}

	// Cut until the text is consumed, rather than stopping at the plan's predicted count.
	//
	// The count is a prediction: cuts land on boundaries, so chunks come in under the target and
	// the shortfall accumulates. Stopping at the predicted number forces the remainder into the
	// last chunk, which can then exceed what a stream will hold — a valid split rejected because
	// the planner's arithmetic assumed exact-sized pieces. Producing one extra chunk is harmless;
	// producing one that does not fit is not.
	limit := plan.MaxChunkTokens
	if limit <= 0 {
		limit = plan.ChunkTokens
	}

	var out []Chunk
	rest := text
	consumed := 0
	for len(strings.TrimSpace(rest)) > 0 {
		var piece string
		quality := CutEnd
		if count(rest) <= plan.ChunkTokens {
			piece, rest = rest, ""
		} else {
			piece, rest, quality = cutAt(rest, plan.ChunkTokens, count,
				localBounds(bounds, consumed, len(rest)))
			if piece == "" {
				return nil, fmt.Errorf("vcontext: could not cut a chunk of %d tokens",
					plan.ChunkTokens)
			}
		}
		if n := count(piece); n > limit {
			return nil, fmt.Errorf("vcontext: a chunk holds %d tokens against a %d limit per "+
				"stream", n, limit)
		}
		start := consumed
		consumed = len(text) - len(rest)
		out = append(out, Chunk{
			Index: len(out), Text: piece, Tokens: count(piece),
			Start: start, End: start + len(piece), Cut: quality,
		})
		// A cut that consumes nothing would loop forever; treat it as a fault rather than hang.
		if len(out) > 1 && out[len(out)-1].Start == out[len(out)-2].Start {
			return nil, fmt.Errorf("vcontext: splitting made no progress at chunk %d", len(out))
		}
	}

	// Fold a runt tail back into its predecessor.
	//
	// A cut near the end can leave a fragment too small to carry a fact — measured: a 21-token
	// tail produced a digest with the wrong port, because the sentence naming it had been left in
	// the previous chunk. A fragment is worse than a slightly oversized neighbour: it costs a
	// stream, and it answers confidently from too little context.
	//
	// Only folded when the result still fits, so this can never create the overflow the loop
	// above exists to avoid.
	if n := len(out); n > 1 {
		last, prev := out[n-1], out[n-2]
		if last.Tokens < plan.ChunkTokens/4 {
			joined := text[prev.Start:last.End]
			if count(joined) <= limit {
				// Fits as one chunk: fold, and save a stream.
				out = out[:n-1]
				out[n-2] = Chunk{
					Index: prev.Index, Text: joined, Tokens: count(joined),
					Start: prev.Start, End: last.End, Cut: last.Cut,
				}
			} else {
				// Too large to merge, so rebalance instead: re-cut the pair at half their
				// combined size. Neither half can then be a runt, and neither can overflow,
				// since each is about half of something that was at most two chunks.
				// Boundaries translated into the joined pair's frame, so a rebalance can
				// still land on a declaration rather than falling back to prose.
				head, tail, q := cutAt(joined, count(joined)/2, count,
					localBounds(bounds, prev.Start, len(joined)))
				if head != "" && strings.TrimSpace(tail) != "" &&
					count(head) <= limit && count(tail) <= limit {
					hStart := prev.Start
					tStart := prev.Start + len(joined) - len(tail)
					out[n-2] = Chunk{
						Index: prev.Index, Text: head, Tokens: count(head),
						Start: hStart, End: hStart + len(head), Cut: q,
					}
					out[n-1] = Chunk{
						Index: last.Index, Text: tail, Tokens: count(tail),
						Start: tStart, End: tStart + len(tail), Cut: last.Cut,
					}
				}
			}
		}
	}

	return out, nil
}

// cutAt takes a prefix of about target tokens, preferring a clean boundary, and returns it with
// the remainder.
// localBounds translates absolute boundary offsets into the remainder's frame, keeping only those
// that fall inside it.
//
// Necessary because the cut loop works on a shrinking remainder while a parser reports offsets into
// the whole file. Getting this wrong would not error — it would cut at a position that means
// nothing, which is the prose behaviour wearing a structural label.
func localBounds(abs []int, consumed, restLen int) []int {
	if len(abs) == 0 {
		return nil
	}
	var out []int
	for _, a := range abs {
		l := a - consumed
		if l > 0 && l <= restLen {
			out = append(out, l)
		}
	}
	return out
}

// lastBoundWithin returns the largest boundary at or before lo and within window characters of it.
func lastBoundWithin(bounds []int, lo, window int) int {
	best := 0
	from := lo - window
	for _, b := range bounds {
		if b <= lo && b >= from && b > best {
			best = b
		}
	}
	return best
}

func cutAt(s string, target int, count CountTokens, bounds []int) (string, string, CutQuality) {
	if count(s) <= target {
		return s, "", CutEnd
	}

	// Binary search the character offset whose token count lands on target. Tokens per character
	// varies by content, so this is measured rather than assumed.
	lo, hi := 0, len(s)
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if count(s[:mid]) <= target {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	if lo == 0 {
		// A single token longer than the whole target: take one character so progress is made
		// rather than looping forever.
		return s[:1], s[1:], CutHard
	}

	// Back up to the best boundary within the last 20% of the chunk. Losing a fifth of a chunk's
	// capacity is cheaper than cutting a sentence in half, because a chunk that ends mid-clause
	// produces an answer about a fragment.
	window := lo / 5
	// A parser-supplied boundary is preferred over every prose heuristic: it is the one cut that
	// can promise a declaration was not severed.
	if b := lastBoundWithin(bounds, lo, window); b > 0 {
		return s[:b], strings.TrimLeft(s[b:], "\n"), CutDeclaration
	}
	if b := lastIndexWithin(s[:lo], window, "\n\n"); b > 0 {
		return s[:b], strings.TrimLeft(s[b:], "\n"), CutParagraph
	}
	if b := lastSentenceEnd(s[:lo], window); b > 0 {
		return s[:b], strings.TrimLeft(s[b:], " "), CutSentence
	}
	if b := lastIndexWithin(s[:lo], window, " "); b > 0 {
		return s[:b], strings.TrimLeft(s[b:], " "), CutWord
	}
	return s[:lo], s[lo:], CutHard
}

// lastIndexWithin finds sep in the final window characters of s, returning the offset just past
// it, or 0 when it is not there.
func lastIndexWithin(s string, window int, sep string) int {
	if window <= 0 || window > len(s) {
		window = len(s)
	}
	from := len(s) - window
	i := strings.LastIndex(s[from:], sep)
	if i < 0 {
		return 0
	}
	return from + i + len(sep)
}

// lastSentenceEnd finds the last sentence terminator in the final window characters.
func lastSentenceEnd(s string, window int) int {
	best := 0
	for _, sep := range []string{". ", ".\n", "! ", "? ", ".\t"} {
		if i := lastIndexWithin(s, window, sep); i > best {
			best = i
		}
	}
	return best
}
