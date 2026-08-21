// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package enginetest provides a deterministic engine.Backend for tests in other packages.
//
// It lives outside internal/engine so that test scaffolding is not linked into the server
// binary, and so the engine's own tests can keep their private fake without an import cycle.
package enginetest

import (
	"sync"

	"github.com/sideblank/llama-herd/internal/engine"
)

// Scripted emits a fixed string, one character per token, for every sequence. It never
// touches a GPU, so tests using it run anywhere.
type Scripted struct {
	mu sync.Mutex

	nCtx     uint32
	nCtxSeq  uint32
	nSeqMax  uint32
	batchCap int32

	script  []engine.Token
	emitted map[engine.SeqID]int
	staged  int32

	eos engine.Token
}

var _ engine.Backend = (*Scripted)(nil)

// New returns a backend that emits script, character by character, for each sequence.
func New(nSeqMax uint32, batchCap int32, script string) *Scripted {
	toks := make([]engine.Token, 0, len(script))
	for _, r := range script {
		toks = append(toks, engine.Token(r))
	}
	return &Scripted{
		nCtx:     4096,
		nCtxSeq:  1024,
		nSeqMax:  nSeqMax,
		batchCap: batchCap,
		script:   toks,
		emitted:  map[engine.SeqID]int{},
		// Far outside the range of any character token, so a scripted letter can
		// never be mistaken for end-of-sequence.
		eos: 1 << 20,
	}
}

func (s *Scripted) NCtx() uint32      { return s.nCtx }
func (s *Scripted) NCtxSeq() uint32   { return s.nCtxSeq }
func (s *Scripted) NSeqMax() uint32   { return s.nSeqMax }
func (s *Scripted) BatchCap() int32   { return s.batchCap }
func (s *Scripted) BatchLen() int32   { return s.staged }
func (s *Scripted) BatchClear()       { s.staged = 0 }
func (s *Scripted) EOS() engine.Token { return s.eos }

// TrimSeq is accepted and ignored: the script is positional, not stateful.
func (s *Scripted) TrimSeq(_ engine.SeqID, _ engine.Pos) {}

func (s *Scripted) FreeSeq(seq engine.SeqID) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.emitted, seq)
}

func (s *Scripted) Tokenize(text string, _ bool) ([]engine.Token, error) {
	out := make([]engine.Token, 0, len(text))
	for _, r := range text {
		out = append(out, engine.Token(r))
	}
	return out, nil
}

func (s *Scripted) Piece(t engine.Token) ([]byte, error) {
	if t == s.eos {
		return nil, nil
	}
	return []byte(string(rune(t))), nil
}

func (s *Scripted) BatchAdd(_ engine.Token, _ engine.Pos, _ engine.SeqID, _ bool) error {
	if s.staged >= s.batchCap {
		return engine.ErrBatchFull
	}
	s.staged++
	return nil
}

func (s *Scripted) Decode() error { return nil }

// SetSampling is accepted and ignored: the script is deterministic by design, so sampling
// settings cannot change what it emits.
func (s *Scripted) SetSampling(_ engine.SeqID, _ *engine.SamplingParams) error { return nil }

func (s *Scripted) SampleAt(seq engine.SeqID, _ int32) (engine.Token, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := s.emitted[seq]
	s.emitted[seq] = i + 1
	if i < len(s.script) {
		return s.script[i], nil
	}
	return s.eos, nil
}

// WithMedia wraps a Scripted backend so it also satisfies engine.MediaBackend, for testing
// the media path without a projector.
type WithMedia struct {
	*Scripted
	// Marker is the placeholder a prompt must contain.
	MarkerText string
	// Prefills records each media prefill: sequence, item count, and prompt.
	Prefills []MediaCall
	// PrefillErr, when set, is returned instead of prefilling.
	PrefillErr error
	// PosAfter is the position reported back to the scheduler.
	PosAfter engine.Pos
}

// MediaCall is one recorded prefill.
type MediaCall struct {
	Seq    engine.SeqID
	Items  int
	Prompt string
}

var _ engine.MediaBackend = (*WithMedia)(nil)

// NewWithMedia returns a scripted backend that accepts media.
func NewWithMedia(nSeqMax uint32, batchCap int32, script string) *WithMedia {
	return &WithMedia{
		Scripted:   New(nSeqMax, batchCap, script),
		MarkerText: "<__media__>",
		PosAfter:   64,
	}
}

func (w *WithMedia) MediaMarker() string { return w.MarkerText }

func (w *WithMedia) PrefillMedia(seq engine.SeqID, _ engine.Pos, prompt string,
	media [][]byte, _ bool) (engine.Pos, error) {
	if w.PrefillErr != nil {
		return 0, w.PrefillErr
	}
	w.mu.Lock()
	w.Prefills = append(w.Prefills, MediaCall{Seq: seq, Items: len(media), Prompt: prompt})
	w.mu.Unlock()
	return w.PosAfter, nil
}
