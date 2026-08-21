// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package draft provides sources of speculative tokens.
package draft

import (
	"sync"

	"github.com/sideblank/llama-herd/internal/engine"
)

// Lookup drafts by finding where the recent tokens appeared before and proposing whatever
// followed them.
//
// It costs no VRAM and needs no second model, which makes it the cheapest speculation
// available: the only price is batch space for proposals that may be rejected, and a rejected
// proposal still yields the target's own token.
//
// It works when output repeats context, which is common in the workloads this engine is built
// for — editing a file that is in the prompt, filling a schema, continuing a transcript,
// emitting structured formats. It contributes nothing to free-form prose, where it simply
// finds no match and proposes nothing.
type Lookup struct {
	// N is the pattern length matched against history. Longer patterns match less often
	// but predict more reliably; shorter ones propose constantly and are usually wrong,
	// wasting batch space.
	N int
	// Max is the most tokens proposed at once.
	Max int
	// MaxHistory bounds retained tokens per sequence. Long contexts are the point, so
	// this is generous, but unbounded growth on a long-running stream is not acceptable.
	MaxHistory int

	mu   sync.Mutex
	hist map[engine.SeqID][]engine.Token
	// last holds the most recent proposal per sequence, so Accept can append the part
	// that was kept. Without it, accepted tokens never enter history: the engine only
	// reports the token preceding the next draft, so any token accepted from a proposal
	// is skipped, and history drifts further from the real output the longer a stream
	// runs — quietly reducing the hit rate exactly when context is richest.
	last map[engine.SeqID][]engine.Token
}

// NewLookup returns a drafter with sensible defaults.
//
// N=3 is the usual compromise: two-token patterns fire on almost anything and are mostly
// wrong, while four or more rarely fire outside heavily repeated text.
func NewLookup(max int) *Lookup {
	if max < 1 {
		max = 4
	}
	return &Lookup{
		N:          3,
		Max:        max,
		MaxHistory: 1 << 20,
		hist:       map[engine.SeqID][]engine.Token{},
		last:       map[engine.SeqID][]engine.Token{},
	}
}

var (
	_ engine.Drafter = (*Lookup)(nil)
	_ engine.Seeder  = (*Lookup)(nil)
)

// MaxDraft bounds proposals per step.
func (l *Lookup) MaxDraft() int { return l.Max }

// Seed records the prompt, which is where most matches come from.
func (l *Lookup) Seed(seq engine.SeqID, tokens []engine.Token) {
	l.mu.Lock()
	defer l.mu.Unlock()
	h := make([]engine.Token, len(tokens))
	copy(h, tokens)
	l.hist[seq] = l.trim(h)
	delete(l.last, seq)
}

// Release drops a finished sequence's history.
func (l *Lookup) Release(seq engine.SeqID) {
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.hist, seq)
	delete(l.last, seq)
}

func (l *Lookup) trim(h []engine.Token) []engine.Token {
	if l.MaxHistory > 0 && len(h) > l.MaxHistory {
		return h[len(h)-l.MaxHistory:]
	}
	return h
}

// Draft proposes the continuation that followed the most recent earlier occurrence of the
// current pattern.
//
// The search runs backwards because recency predicts better: in a file being edited, the
// nearest previous occurrence of a line is far more likely to continue the same way than one
// from thousands of tokens earlier.
func (l *Lookup) Draft(seq engine.SeqID, last engine.Token, _ engine.Pos, n int) ([]engine.Token, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	h := append(l.hist[seq], last)
	l.hist[seq] = l.trim(h)
	h = l.hist[seq]

	if n > l.Max {
		n = l.Max
	}
	if n < 1 || len(h) < l.N+1 {
		return nil, nil
	}

	pattern := h[len(h)-l.N:]
	// Stop before the pattern's own position so it cannot match itself.
	for i := len(h) - l.N - 1; i >= 0; i-- {
		if !equal(h[i:i+l.N], pattern) {
			continue
		}
		start := i + l.N
		end := start + n
		if end > len(h) {
			end = len(h)
		}
		if start >= end {
			return nil, nil
		}
		out := make([]engine.Token, end-start)
		copy(out, h[start:end])
		l.last[seq] = out
		return out, nil
	}
	l.last[seq] = nil
	return nil, nil
}

// Accept records which proposals the target kept, so history matches what was really emitted.
//
// This has to be exact. History that includes rejected tokens would have the drafter matching
// against text the model never produced, and its proposals would drift further wrong the
// longer a stream ran.
func (l *Lookup) Accept(seq engine.SeqID, accepted int, corrected engine.Token) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Accepted proposals really were emitted, so they belong in history. The engine only
	// hands back the token preceding the next draft, so without appending them here they
	// would never be recorded at all.
	if n := accepted; n > 0 {
		if p := l.last[seq]; len(p) >= n {
			l.hist[seq] = l.trim(append(l.hist[seq], p[:n]...))
		}
	}
	l.last[seq] = nil
	_ = corrected
	return nil
}

// Observe appends tokens the engine actually emitted. Callers that can report real output
// should use it, since history accuracy is what makes the drafter worth anything.
func (l *Lookup) Observe(seq engine.SeqID, tokens ...engine.Token) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.hist[seq] = l.trim(append(l.hist[seq], tokens...))
}

func equal(a, b []engine.Token) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
