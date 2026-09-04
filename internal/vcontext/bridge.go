// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"fmt"
	"sort"
)

// Bridge is a window spanning material that chunking separated.
//
// This is what the spare stream capacity is for. Splitting a document is otherwise lossy in one
// specific way that nothing downstream can repair: a fact in one chunk relating to a fact in
// another is seen by no stream, so no index entry records the relationship and no selection can
// retrieve it. Bridges are the streams that do see both.
//
// They are free in wall-clock terms only if they ride the SAME batch as the chunks. A bridge that
// spills into a second wave costs a full pass, so the allocator fits them inside the stream budget
// rather than adding to it.
type Bridge struct {
	// Left and Right are the chunks this window spans.
	Left, Right int
	// Start and End are byte offsets into the original input. For an adjacent bridge this is a
	// contiguous window across the cut; for a long-range pair the two halves are disjoint and
	// Text assembles them with a marker.
	Start, End int
	// RightStart and RightEnd are set only for long-range pairs, where the two regions are not
	// contiguous.
	RightStart, RightEnd int
	// Adjacent distinguishes a window straddling one cut from a pairing of distant chunks.
	Adjacent bool
	// Priority is why this bridge was chosen; higher is more urgent. Recorded so a decision to
	// drop bridges under pressure can be reviewed.
	Priority float64
	// Because states the reason in words, for the same purpose.
	Because string
}

// Text returns the bridge's source, joining the halves of a long-range pair with a marker so the
// model is not misled into reading two distant regions as contiguous.
func (b Bridge) Text(source string) (string, error) {
	if b.Start < 0 || b.End > len(source) || b.Start > b.End {
		return "", fmt.Errorf("vcontext: bridge [%d,%d) is outside a source of %d bytes",
			b.Start, b.End, len(source))
	}
	left := source[b.Start:b.End]
	if b.Adjacent {
		return left, nil
	}
	if b.RightStart < 0 || b.RightEnd > len(source) || b.RightStart > b.RightEnd {
		return "", fmt.Errorf("vcontext: bridge right half [%d,%d) is outside the source",
			b.RightStart, b.RightEnd)
	}
	return left + "\n\n[... distant passage, not contiguous with the above ...]\n\n" +
		source[b.RightStart:b.RightEnd], nil
}

// StreamBudget is how the deployment's streams are divided for one request.
type StreamBudget struct {
	// Total is what the deployment can run concurrently in one batch.
	Total int
	// Chunks is how many carry the user's input.
	Chunks int
	// Spare is what is left, and is what bridges may use.
	Spare int
}

// Budget divides the available streams between the input and the spare capacity above it.
//
// The spare exists because the deployment allocates more context than a caller may send: 48
// streams of 8,874 is ~426k against a 256k ceiling, so roughly 20 streams are already allocated,
// already paid for, and idle unless something is given to them. Anything they do concurrently
// with the chunk pass adds no wall-clock time, because they ride the same forward pass.
func Budget(total, chunks int) (StreamBudget, error) {
	if total < 1 {
		return StreamBudget{}, fmt.Errorf("vcontext: total streams must be positive, got %d", total)
	}
	if chunks < 1 || chunks > total {
		return StreamBudget{}, fmt.Errorf("vcontext: %d chunks does not fit %d streams",
			chunks, total)
	}
	return StreamBudget{Total: total, Chunks: chunks, Spare: total - chunks}, nil
}

// PlanBridges chooses which separations to repair, given the spare capacity.
//
// Adjacent boundaries come first and are ordered by how badly they were cut, because the splitter
// already knows: a chunk ending at a paragraph break probably severed nothing, one ending mid-word
// certainly did. That signal is free and it is a far better ranking than position.
//
// Long-range pairs follow, ordered by shared topics, because a relationship between distant chunks
// is only worth a stream when there is a reason to suspect one. Pairing every chunk with every
// other is 496 pairs at 32 chunks; the index makes it a handful.
func PlanBridges(chunks []Chunk, budget StreamBudget, window int, related func(a, b Chunk) float64) []Bridge {
	if budget.Spare <= 0 || len(chunks) < 2 {
		return nil
	}
	if window <= 0 {
		window = 512
	}

	var out []Bridge

	// Adjacent: a window straddling each cut, worst cuts first.
	for i := 0; i+1 < len(chunks); i++ {
		l, r := chunks[i], chunks[i+1]
		start := l.End - window
		if start < l.Start {
			start = l.Start
		}
		end := r.Start + window
		if end > r.End {
			end = r.End
		}
		out = append(out, Bridge{
			Left: l.Index, Right: r.Index,
			Start: start, End: end, Adjacent: true,
			// A worse cut severed more, so it needs a bridge more badly. Compared by
			// ordinal, so inserting a new quality tier does not change the ranking.
			Priority: float64(l.Cut),
			Because:  fmt.Sprintf("chunk %d ended on a %s boundary", l.Index, l.Cut),
		})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Priority > out[j].Priority })

	// Long-range: only where the index suggests a relationship.
	if related != nil {
		var far []Bridge
		for i := 0; i < len(chunks); i++ {
			for j := i + 2; j < len(chunks); j++ {
				score := related(chunks[i], chunks[j])
				if score <= 0 {
					continue
				}
				li, rj := chunks[i], chunks[j]
				ls := li.End - window
				if ls < li.Start {
					ls = li.Start
				}
				re := rj.Start + window
				if re > rj.End {
					re = rj.End
				}
				far = append(far, Bridge{
					Left: li.Index, Right: rj.Index,
					Start: ls, End: li.End,
					RightStart: rj.Start, RightEnd: re,
					Priority: score,
					Because: fmt.Sprintf("chunks %d and %d share material (score %.2f)",
						li.Index, rj.Index, score),
				})
			}
		}
		sort.SliceStable(far, func(i, j int) bool { return far[i].Priority > far[j].Priority })
		out = append(out, far...)
	}

	// Never exceed the spare capacity. A bridge that spills into a second wave costs a full
	// pass, which is the opposite of the point.
	if len(out) > budget.Spare {
		out = out[:budget.Spare]
	}
	return out
}

// UnbridgedCuts reports boundaries that were severed and left unrepaired, worst first.
//
// The caller needs this. It is the difference between "the document does not relate those things"
// and "we did not have a stream to look". A hard cut left unbridged is a known blind spot, and a
// system that cannot name its blind spots cannot be trusted to say a document is silent on
// something.
func UnbridgedCuts(chunks []Chunk, planned []Bridge) []Chunk {
	covered := map[int]bool{}
	for _, b := range planned {
		if b.Adjacent {
			covered[b.Left] = true
		}
	}
	var out []Chunk
	for i := 0; i+1 < len(chunks); i++ {
		if !covered[chunks[i].Index] && chunks[i].Cut >= CutWord {
			out = append(out, chunks[i])
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Cut > out[j].Cut })
	return out
}
