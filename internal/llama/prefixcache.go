// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

import (
	"fmt"

	"github.com/sideblank/llama-herd/internal/engine"

	"github.com/sideblank/llama-herd/internal/prefix"
)

// PrefixCache adapts a Runner to the prefix package's Cache interface, so a shared prompt prefix
// can be computed once and its KV cells shared across the herd.
//
// NOT SAFE FOR CONCURRENT USE, for the same reason as Runner: it drives the one context, and every
// method here must be called from the decode-loop goroutine.
type PrefixCache struct {
	r *Runner
}

// NewPrefixCache wraps a runner. The runner must own the context nothing else is touching.
func NewPrefixCache(r *Runner) *PrefixCache { return &PrefixCache{r: r} }

// Unified reports whether the context has one shared attention buffer.
func (p *PrefixCache) Unified() bool { return p.r.ctx.KVUnified() }

// DecodeSeq evaluates tokens for one sequence starting at position from.
//
// Chunked to the batch capacity: a shared header is routinely larger than one batch, and a prefix
// long enough to be worth sharing is exactly the case where that happens.
//
// No logits are requested for any token. The prefix is being cached, not sampled from — asking for
// logits on tokens nobody reads spends output-buffer space that the real streams need.
func (p *PrefixCache) DecodeSeq(seq int, tokens []prefix.Token, from int) error {
	capacity := int(p.r.batch.Cap())
	if capacity <= 0 {
		return fmt.Errorf("llama: the runner's batch has no capacity")
	}
	for off := 0; off < len(tokens); off += capacity {
		end := off + capacity
		if end > len(tokens) {
			end = len(tokens)
		}
		p.r.batch.Clear()
		for i := off; i < end; i++ {
			pos := Pos(from + i)
			if err := p.r.batch.Add(Token(tokens[i]), pos, []SeqID{SeqID(seq)}, false); err != nil {
				return fmt.Errorf("llama: building the prefix batch at %d: %w", i, err)
			}
		}
		if err := p.r.ctx.Decode(p.r.batch); err != nil {
			return fmt.Errorf("llama: decoding the prefix at %d..%d: %w", off, end, err)
		}
	}
	return nil
}

// CopyPrefix shares src's cells in [0, n) with dst.
func (p *PrefixCache) CopyPrefix(src, dst, n int) error {
	return p.r.ctx.SeqCp(SeqID(src), SeqID(dst), 0, Pos(n))
}

// PosMax returns the highest cached position for a sequence, or -1 if it holds nothing.
func (p *PrefixCache) PosMax(seq int) int { return int(p.r.ctx.SeqPosMax(SeqID(seq))) }

var _ prefix.Cache = (*PrefixCache)(nil)

// SharePrefix gives dst the cells src holds in [0, n), satisfying engine.PrefixSharer.
//
// Called only from the decode-loop goroutine, like every other method that touches the context.
func (r *Runner) SharePrefix(src, dst engine.SeqID, n int) error {
	return r.ctx.SeqCp(SeqID(src), SeqID(dst), 0, Pos(n))
}

// CachedThrough reports the highest position a sequence holds, or -1 for none.
func (r *Runner) CachedThrough(seq engine.SeqID) int {
	return int(r.ctx.SeqPosMax(SeqID(seq)))
}

var _ engine.PrefixSharer = (*Runner)(nil)
