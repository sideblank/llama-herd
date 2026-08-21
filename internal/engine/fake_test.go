// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package engine

import (
	"fmt"
	"sync"
)

// fakeBackend is a deterministic stand-in for libllama. Each sequence emits a fixed script
// of tokens, so the scheduler's behaviour is fully observable without a GPU.
type fakeBackend struct {
	mu sync.Mutex

	nCtx     uint32
	nCtxSeq  uint32
	nSeqMax  uint32
	batchCap int32

	// staged holds the entries added since BatchClear.
	staged []staged
	// script is the token each sequence emits, by call count.
	script map[SeqID][]Token
	// emitted counts sampled tokens per sequence.
	emitted map[SeqID]int

	eos Token

	// decodeErr, when set, is returned by the next Decode.
	decodeErr error

	// observed records the batch size of each Decode, for budget assertions.
	observed []int32
	// maxSeen is the largest batch ever staged.
	maxSeen int32
	// freed counts FreeSeq calls per sequence.
	freed map[SeqID]int
	// trims records TrimSeq calls.
	trims []trim
	// sampling records the last params installed per sequence.
	sampling map[SeqID]*SamplingParams
	// samplingCalls counts SetSampling calls per sequence.
	samplingCalls map[SeqID]int
}

type trim struct {
	seq  SeqID
	from Pos
}

type staged struct {
	tok        Token
	pos        Pos
	seq        SeqID
	wantLogits bool
}

func newFake(nSeqMax uint32, batchCap int32) *fakeBackend {
	return &fakeBackend{
		nCtx:          4096,
		nCtxSeq:       1024,
		nSeqMax:       nSeqMax,
		batchCap:      batchCap,
		script:        map[SeqID][]Token{},
		emitted:       map[SeqID]int{},
		freed:         map[SeqID]int{},
		sampling:      map[SeqID]*SamplingParams{},
		samplingCalls: map[SeqID]int{},
		// Deliberately outside the range of any scripted character token: an EOS of 99
		// would collide with ASCII 'c' and end generation whenever that letter appeared.
		eos: 1 << 20,
	}
}

func (f *fakeBackend) NCtx() uint32    { return f.nCtx }
func (f *fakeBackend) NCtxSeq() uint32 { return f.nCtxSeq }
func (f *fakeBackend) NSeqMax() uint32 { return f.nSeqMax }
func (f *fakeBackend) BatchCap() int32 { return f.batchCap }
func (f *fakeBackend) EOS() Token      { return f.eos }
func (f *fakeBackend) BatchLen() int32 { return int32(len(f.staged)) }
func (f *fakeBackend) BatchClear()     { f.staged = f.staged[:0] }

func (f *fakeBackend) Tokenize(text string, _ bool) ([]Token, error) {
	if text == "" {
		return nil, nil
	}
	out := make([]Token, 0, len(text))
	for _, r := range text {
		out = append(out, Token(r))
	}
	return out, nil
}

func (f *fakeBackend) Piece(t Token) ([]byte, error) {
	if t == f.eos {
		return nil, nil
	}
	return []byte(string(rune(t))), nil
}

func (f *fakeBackend) BatchAdd(tok Token, pos Pos, seq SeqID, wantLogits bool) error {
	if int32(len(f.staged)) >= f.batchCap {
		return ErrBatchFull
	}
	f.staged = append(f.staged, staged{tok, pos, seq, wantLogits})
	return nil
}

func (f *fakeBackend) Decode() error {
	n := int32(len(f.staged))
	f.observed = append(f.observed, n)
	if n > f.maxSeen {
		f.maxSeen = n
	}
	if n > f.batchCap {
		return fmt.Errorf("fake: batch of %d exceeds cap %d", n, f.batchCap)
	}
	if err := f.decodeErr; err != nil {
		f.decodeErr = nil
		return err
	}
	return nil
}

func (f *fakeBackend) SampleAt(seq SeqID, _ int32) (Token, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	i := f.emitted[seq]
	f.emitted[seq] = i + 1
	if s := f.script[seq]; i < len(s) {
		return s[i], nil
	}
	return f.eos, nil
}

func (f *fakeBackend) SetSampling(seq SeqID, p *SamplingParams) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.sampling[seq] = p
	f.samplingCalls[seq]++
	return nil
}

// trimmed records TrimSeq calls so tests can assert rejected drafts are rolled back.
func (f *fakeBackend) TrimSeq(seq SeqID, from Pos) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.trims = append(f.trims, trim{seq, from})
}

func (f *fakeBackend) FreeSeq(seq SeqID) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.freed[seq]++
}
