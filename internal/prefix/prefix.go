// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package prefix computes a prompt prefix once and shares its KV cells across the herd.
//
// The case this exists for: a wide fan-out where every stream carries the same leading tokens — a
// global symbol table, a schema header, a system prompt. At 48 streams a 1.5k-token header is
// 72,000 tokens of prefill spent re-reading identical bytes, which measured against the shared
// prefill rate is most of a minute and 28% of a 256k payload.
//
// With a unified KV cache the fix is nearly free. Upstream marks each cell as belonging to the
// destination sequence as well — no data is copied — so the prefix is computed once and every
// stream attends to the same cells. That makes it a memory saving as well as a compute one: the
// shared cells are held once rather than N times.
package prefix

import (
	"errors"
	"fmt"
)

// Token is a vocabulary index. Declared here so the planning half needs no cgo and can be tested
// without a GPU.
type Token = int32

// Plan describes how a batch of prompts should share work.
type Plan struct {
	// Donor is the index of the sequence that computes the shared prefix.
	Donor int
	// Prefix is the identical leading run of tokens, computed once by the donor.
	Prefix []Token
	// Suffixes are the per-prompt tokens still to be decoded, in the caller's order.
	Suffixes [][]Token
	// Saved is the number of prefill tokens this avoids: SharedLen x (n-1).
	Saved int
	// Total is what the batch would have cost with no sharing.
	Total int
}

// SharedLen is how many leading tokens are shared.
func (p Plan) SharedLen() int { return len(p.Prefix) }

// Fraction is the share of prefill this plan avoids.
func (p Plan) Fraction() float64 {
	if p.Total == 0 {
		return 0
	}
	return float64(p.Saved) / float64(p.Total)
}

// Worth reports whether the plan clears a minimum shared length.
//
// Sharing is not free: upstream walks every cell in the cache once per destination sequence to
// re-tag it. Below some prefix length that bookkeeping costs more than the prefill it avoids, and
// where that line falls is a property of the deployment. It is a parameter rather than a constant
// because it has not been measured yet — see #37.
func (p Plan) Worth(minShared int) bool {
	return len(p.Prefix) >= minShared && len(p.Suffixes) > 1
}

// ErrNoPrompts is returned for an empty batch.
var ErrNoPrompts = errors.New("prefix: no prompts")

// Analyse finds the longest common token prefix across prompts and plans the split.
//
// ⛔ Computed on TOKENS, never on text. A common *string* prefix does not imply a common *token*
// prefix: BPE merges across the boundary, so two prompts sharing the characters "user_" can
// tokenise to different tokens there depending on what follows. Copying cells for a prefix derived
// from text would hand a sequence KV entries for tokens it does not actually have, and the model
// would attend to the wrong history with nothing to signal it.
func Analyse(prompts [][]Token) (Plan, error) {
	if len(prompts) == 0 {
		return Plan{}, ErrNoPrompts
	}
	total := 0
	shortest := len(prompts[0])
	for _, p := range prompts {
		total += len(p)
		if len(p) < shortest {
			shortest = len(p)
		}
	}
	if len(prompts) == 1 {
		return Plan{Donor: 0, Suffixes: prompts, Total: total}, nil
	}

	// A sequence must keep at least one token to decode: llama produces logits from tokens it is
	// given this pass, so a sequence whose whole prompt was absorbed into a shared prefix would
	// have nothing to evaluate and no logits to sample from.
	limit := shortest - 1
	if limit < 0 {
		limit = 0
	}

	n := 0
	for n < limit {
		t := prompts[0][n]
		same := true
		for _, p := range prompts[1:] {
			if p[n] != t {
				same = false
				break
			}
		}
		if !same {
			break
		}
		n++
	}

	suffixes := make([][]Token, len(prompts))
	for i, p := range prompts {
		suffixes[i] = p[n:]
	}
	return Plan{
		Donor:    0,
		Prefix:   prompts[0][:n:n],
		Suffixes: suffixes,
		Saved:    n * (len(prompts) - 1),
		Total:    total,
	}, nil
}

// Cache is the subset of a llama context this package needs.
//
// An interface so the sharing logic is testable without a GPU, and so the verification below can be
// exercised against a cache that deliberately misbehaves — which matters, because the real one has
// a silent-no-op path.
type Cache interface {
	// Unified reports whether the KV cache is one shared buffer. Partial copies require it.
	Unified() bool
	// DecodeSeq evaluates tokens for a sequence starting at position from.
	DecodeSeq(seq int, tokens []Token, from int) error
	// CopyPrefix shares src's cells in [0, n) with dst.
	CopyPrefix(src, dst, n int) error
	// PosMax returns the highest position held for a sequence, or -1 for none.
	PosMax(seq int) int
}

// ErrNotUnified is returned when sharing is attempted without a unified cache.
var ErrNotUnified = errors.New("prefix: sharing a partial prefix requires a unified KV cache")

// Share computes the shared prefix once on the donor sequence and copies its cells to the others.
//
// seqs maps each prompt index to the sequence id it should occupy.
//
// The verification after each copy is not defensive padding. `llama_memory_seq_cp` returns void, has
// an early-return path that does nothing at all, and asserts rather than errors on a partial copy
// without a unified cache. A sequence left with an empty cache generates fluent output grounded in
// no history — the failure would surface as a bad answer, not as an error.
func Share(c Cache, plan Plan, seqs []int) error {
	if len(seqs) != len(plan.Suffixes) {
		return fmt.Errorf("prefix: %d sequence ids for %d prompts", len(seqs), len(plan.Suffixes))
	}
	n := len(plan.Prefix)
	if n == 0 {
		return nil
	}
	if !c.Unified() {
		return ErrNotUnified
	}

	donorSeq := seqs[plan.Donor]
	if err := c.DecodeSeq(donorSeq, plan.Prefix, 0); err != nil {
		return fmt.Errorf("prefix: computing the shared prefix on seq %d: %w", donorSeq, err)
	}
	if got := c.PosMax(donorSeq); got != n-1 {
		return fmt.Errorf("prefix: the donor holds %d positions after decoding %d tokens — the "+
			"prefix was not cached, so copying it would share nothing", got+1, n)
	}

	for i, seq := range seqs {
		if i == plan.Donor {
			continue
		}
		if err := c.CopyPrefix(donorSeq, seq, n); err != nil {
			return fmt.Errorf("prefix: sharing to seq %d: %w", seq, err)
		}
		if got := c.PosMax(seq); got != n-1 {
			return fmt.Errorf("prefix: seq %d holds %d positions after a copy of %d — the copy "+
				"reported success and did nothing, and this sequence would generate from an "+
				"empty history", seq, got+1, n)
		}
	}
	return nil
}
