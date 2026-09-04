// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package engine

import "sort"

// PrefixSharer is a backend that can hand one sequence's cached prefix to another without
// recomputing it.
//
// Optional: a backend that cannot do this simply does not implement it, and prefill proceeds
// normally. Implemented by the llama.cpp runner over a unified KV cache, where sharing is a
// metadata operation rather than a copy — upstream tags the same cells as belonging to both
// sequences.
type PrefixSharer interface {
	// SharePrefix gives dst the cells src holds in [0, n).
	SharePrefix(src, dst SeqID, n int) error
	// CachedThrough reports the highest position a sequence holds, or -1 for none. It is the
	// only way to confirm a share took effect.
	CachedThrough(seq SeqID) int
}

// minSharedPrefix is the shortest prefix worth sharing.
//
// Sharing is not free: the backend walks its cells once per destination to re-tag them, so below
// some length the bookkeeping costs more than the prefill it avoids. Where that line falls has not
// been measured (#37), so this is deliberately conservative — a wrong guess here wastes a little
// work, while a wrong guess in the other direction would make short requests slower for no reason.
const minSharedPrefix = 256

// sharePrefixes lets freshly admitted slots adopt a prefix another live slot already holds.
//
// Runs before prefill. The case it exists for is a wide fan-out where every stream carries the same
// leading tokens — a system prompt, a global symbol table, a schema header. At 48 streams a 1,500
// token header is 72,000 tokens of prefill spent re-reading identical bytes.
//
// Opportunistic rather than batched, deliberately: requests arrive one at a time through Submit, so
// there is no point at which the whole set is known. Matching against whatever is already resident
// works regardless of arrival order.
//
// ⭐ Safe across donor lifetime. Cells are reference-counted by sequence membership upstream — a
// donor finishing drops only its own tag, and the cell survives while any borrower still holds it —
// so a borrower never needs the donor to stay alive.
func (e *Engine) sharePrefixes(active map[SeqID]*slot) {
	sharer, ok := e.be.(PrefixSharer)
	if !ok {
		return
	}

	for _, s := range active {
		// Only a slot that has not started: sharing into a partly-filled sequence would
		// duplicate positions it already holds.
		if s.primed || s.pos != 0 || len(s.pending) == 0 || s.ctx.Err() != nil {
			continue
		}

		donor, n := bestDonor(s, active)
		if n < minSharedPrefix {
			continue
		}

		if err := sharer.SharePrefix(donor.seq, s.seq, n); err != nil {
			// Falling back to ordinary prefill is always correct — it computes what the
			// share would have copied. Nothing is left half-applied because pos and
			// pending are untouched until the share is verified below.
			e.c.prefixShareFailed.Add(1)
			continue
		}
		// Verify. The backend's underlying call can report nothing and do nothing, and a slot
		// that believes it holds cells it does not would generate fluently from an empty
		// history — a wrong answer rather than an error.
		if got := sharer.CachedThrough(s.seq); got != n-1 {
			e.c.prefixShareFailed.Add(1)
			continue
		}

		s.pos = Pos(n)
		s.pending = s.pending[n:]
		s.sharedPrefix = n
		e.c.prefixShared.Add(1)
		e.c.prefixTokensSaved.Add(uint64(n))
	}
}

// bestDonor finds the live slot sharing the longest cached token prefix with s.
//
// Two constraints that are easy to miss:
//
//   - The donor must have ALREADY prefilled past the shared run. A slot admitted in the same tick
//     holds no cells yet, so copying from it would share nothing while reporting success.
//   - The borrower must keep at least one token to feed. llama produces logits from tokens given
//     this pass, so a prompt entirely absorbed into a shared prefix leaves nothing to sample from.
func bestDonor(s *slot, active map[SeqID]*slot) (*slot, int) {
	var best *slot
	bestN := 0
	// Iterated in sequence order rather than map order: Go randomises map iteration, and two
	// donors offering an equally long prefix would otherwise be chosen arbitrarily, so the same
	// inputs would not produce the same plan twice.
	seqs := make([]SeqID, 0, len(active))
	for id := range active {
		seqs = append(seqs, id)
	}
	sort.Slice(seqs, func(i, j int) bool { return seqs[i] < seqs[j] })
	for _, id := range seqs {
		d := active[id]
		if d == s || len(d.promptToks) == 0 || d.ctx.Err() != nil {
			continue
		}
		n := commonPrefix(s.promptToks, d.promptToks)
		// Cap at what the donor has actually cached, and leave the borrower a token.
		if cached := int(d.pos); n > cached {
			n = cached
		}
		if n >= len(s.promptToks) {
			n = len(s.promptToks) - 1
		}
		if n > bestN {
			best, bestN = d, n
		}
	}
	return best, bestN
}

// commonPrefix counts the leading tokens two prompts share.
//
// ⛔ Compared on TOKENS, never on text. BPE merges across a boundary, so two prompts sharing
// characters can tokenise differently there — cells copied for a text-derived prefix would give a
// sequence history it does not actually have, silently.
func commonPrefix(a, b []Token) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
