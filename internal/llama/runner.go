// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

import (
	"fmt"

	"github.com/sideblank/llama-herd/internal/engine"
)

// Runner is one loaded model driven by one context: the concrete backend the scheduler runs
// against. It satisfies engine.Backend.
//
// NOT SAFE FOR CONCURRENT USE. Every method is expected to be called from the scheduler's
// single decode-loop goroutine, which is what lets it hold a reusable batch and per-sequence
// samplers without locking.
type Runner struct {
	model *Model
	ctx   *Context
	vocab *Vocab
	batch *Batch

	// samplers is one chain per sequence. They must not be shared: a chain carries the
	// penalty window, so two streams on one chain would penalise each other's history.
	samplers []*Sampler

	nVocab int32
	eos    Token
}

// Compile-time proof that Runner satisfies the scheduler's interface. Without this the
// mismatch would only surface where the two are first wired together.
var _ engine.Backend = (*Runner)(nil)

// RunnerConfig describes a model to load and how to serve it.
type RunnerConfig struct {
	ModelPath string
	Model     ModelParams
	Context   ContextParams
	Sampling  SamplingParams
}

// OpenRunner loads a model and prepares it to be scheduled.
func OpenRunner(cfg RunnerConfig) (*Runner, error) {
	m, err := LoadModel(cfg.ModelPath, cfg.Model)
	if err != nil {
		return nil, err
	}

	ctx, err := NewContext(m, cfg.Context)
	if err != nil {
		m.Free()
		return nil, err
	}

	vocab := m.Vocab()
	nVocab := vocab.NTokens()

	nSeq := int(cfg.Context.NSeqMax)
	if nSeq < 1 {
		nSeq = 1
	}

	r := &Runner{
		model:    m,
		ctx:      ctx,
		vocab:    vocab,
		nVocab:   nVocab,
		eos:      vocab.EOS(),
		samplers: make([]*Sampler, nSeq),
	}

	for i := range r.samplers {
		s, err := NewSampler(cfg.Sampling, nVocab)
		if err != nil {
			r.Close()
			return nil, fmt.Errorf("sampler %d: %w", i, err)
		}
		r.samplers[i] = s
	}

	// The batch is sized to the logical batch, which is the ceiling llama_decode accepts
	// in one pass and therefore the scheduler's per-tick budget.
	nBatch := int32(cfg.Context.NBatch)
	if nBatch <= 0 {
		nBatch = int32(ctx.NCtx())
	}
	r.batch = NewBatch(nBatch, int32(nSeq))

	return r, nil
}

// Close releases the batch, samplers, context and weights, in that order.
func (r *Runner) Close() {
	if r.batch != nil {
		r.batch.Free()
		r.batch = nil
	}
	for i, s := range r.samplers {
		s.Free()
		r.samplers[i] = nil
	}
	if r.ctx != nil {
		r.ctx.Free()
		r.ctx = nil
	}
	if r.model != nil {
		r.model.Free()
		r.model = nil
	}
}

// --- engine.Backend ---------------------------------------------------------

func (r *Runner) NCtx() uint32    { return r.ctx.NCtx() }
func (r *Runner) NCtxSeq() uint32 { return r.ctx.NCtxSeq() }
func (r *Runner) NSeqMax() uint32 { return uint32(len(r.samplers)) }
func (r *Runner) BatchCap() int32 { return r.batch.Cap() }
func (r *Runner) BatchLen() int32 { return r.batch.Len() }
func (r *Runner) BatchClear()     { r.batch.Clear() }
func (r *Runner) EOS() engine.Token {
	return engine.Token(r.eos)
}

func (r *Runner) Tokenize(text string, addSpecial bool) ([]engine.Token, error) {
	toks, err := r.vocab.Tokenize(text, addSpecial, true)
	if err != nil {
		return nil, err
	}
	out := make([]engine.Token, len(toks))
	for i, t := range toks {
		out[i] = engine.Token(t)
	}
	return out, nil
}

func (r *Runner) Piece(t engine.Token) ([]byte, error) {
	return r.vocab.TokenToPiece(Token(t), false)
}

func (r *Runner) BatchAdd(tok engine.Token, pos engine.Pos, seq engine.SeqID, wantLogits bool) error {
	err := r.batch.Add(Token(tok), Pos(pos), []SeqID{SeqID(seq)}, wantLogits)
	switch {
	case err == nil:
		return nil
	case err == ErrBatchFull:
		return engine.ErrBatchFull
	default:
		return err
	}
}

func (r *Runner) Decode() error {
	err := r.ctx.Decode(r.batch)
	switch {
	case err == nil:
		return nil
	case err == ErrNoKVSlot:
		// Translate rather than wrap, so the scheduler's recovery path matches on its
		// own sentinel and stays free of any llama.cpp import.
		return engine.ErrNoKVSlot
	default:
		return err
	}
}

func (r *Runner) SampleAt(seq engine.SeqID, i int32) (engine.Token, error) {
	if int(seq) < 0 || int(seq) >= len(r.samplers) {
		return 0, fmt.Errorf("llama: sequence %d out of range (have %d)", seq, len(r.samplers))
	}
	return engine.Token(r.samplers[seq].Sample(r.ctx, i)), nil
}

// FreeSeq drops the sequence's KV cells and clears its sampler.
//
// Resetting the sampler is not optional: the chain holds the penalty window, so a reused
// slot would otherwise carry the previous request's history into an unrelated one.
func (r *Runner) FreeSeq(seq engine.SeqID) {
	r.ctx.SeqRm(SeqID(seq), -1, -1)
	if int(seq) >= 0 && int(seq) < len(r.samplers) {
		r.samplers[seq].Reset()
	}
}
