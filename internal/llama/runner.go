// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

import (
	"fmt"
	"log"

	"github.com/sideblank/llama-herd/internal/bench"
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
	// ckpt holds one partial-state snapshot per sequence, reused across steps so that
	// speculating does not allocate on every draft.
	ckpt [][]byte
	// threads and outputsMax record what the context was actually created with, so startup can
	// report the tuning rather than what it intended.
	threads    int32
	outputsMax uint32
	// thinkPrime opens and closes a reasoning block, suppressing it. Empty when the model
	// has no such block, in which case nothing is ever appended.
	thinkPrime string
	// ckptLogged keeps the snapshot-size report to once per process.
	ckptLogged bool
	// ownsModel records whether Close should free the weights. A sweep holds one copy of the
	// weights across many contexts, and freeing them with the first context would make each
	// configuration pay a fresh load — which is the entire cost the sweep exists to avoid.
	ownsModel bool

	nVocab int32
	eos    Token

	// Recorded at load so the server can report how this model was configured.
	gpuLayers int32
	kvTypeK   string
	kvTypeV   string
	flashAttn bool
	loadMTP   bool

	// defaultSampling is the model's configured sampling, restored whenever a request
	// asks for nothing specific.
	defaultSampling SamplingParams
	// custom marks sequences currently carrying a per-request chain, so a slot is only
	// rebuilt when it actually differs from the default.
	custom []bool

	// vision is non-nil only when a projector was supplied. A text-only model is the
	// common case and must not pay for the multimodal context.
	vision *Vision

	// chatTmpl is captured at load. Rendering then touches no live context state, which
	// is what makes RenderChat safe to call from request goroutines while the decode
	// loop is running.
	chatTmpl string
}

// kvName renders a KV cache type for reporting.
func kvName(t GGMLType) string {
	switch t {
	case TypeQ8_0:
		return "q8_0"
	case TypeQ5_1:
		return "q5_1"
	case TypeQ4_0:
		return "q4_0"
	default:
		return "f16"
	}
}

// Compile-time proof that Runner satisfies the scheduler's interface. Without this the
// mismatch would only surface where the two are first wired together.
var (
	_ engine.Backend          = (*Runner)(nil)
	_ engine.Renderer         = (*Runner)(nil)
	_ engine.Rewinder         = (*Runner)(nil)
	_ engine.ThinkingRenderer = (*Runner)(nil)
	_ bench.PerfSource        = (*Runner)(nil)
)

// RunnerConfig describes a model to load and how to serve it.
type RunnerConfig struct {
	ModelPath string
	Model     ModelParams
	Context   ContextParams
	Sampling  SamplingParams

	// MMProjPath enables the vision lane. Empty leaves the model text-only.
	MMProjPath string
	// VisionGPU offloads the image encoder.
	VisionGPU bool
}

// OpenRunner loads a model and prepares it to be scheduled.
func OpenRunner(cfg RunnerConfig) (*Runner, error) {
	m, err := LoadModel(cfg.ModelPath, cfg.Model)
	if err != nil {
		return nil, err
	}
	r, err := OpenRunnerWithModel(m, cfg)
	if err != nil {
		m.Free()
		return nil, err
	}
	r.ownsModel = true
	return r, nil
}

// OpenRunnerWithModel prepares a runner over weights the caller already loaded and continues to
// own. Closing the runner leaves the model alive.
//
// This is what makes a configuration sweep affordable: loading a 14 GiB model takes longer than
// measuring it, so a sweep that reloaded per configuration would spend nearly all its time on
// the part that does not vary.
func OpenRunnerWithModel(m *Model, cfg RunnerConfig) (*Runner, error) {
	ctx, err := NewContext(m, cfg.Context)
	if err != nil {
		return nil, err
	}

	vocab := m.Vocab()
	nVocab := vocab.NTokens()

	nSeq := int(cfg.Context.NSeqMax)
	if nSeq < 1 {
		nSeq = 1
	}

	r := &Runner{
		model:           m,
		ctx:             ctx,
		vocab:           vocab,
		nVocab:          nVocab,
		eos:             vocab.EOS(),
		gpuLayers:       cfg.Model.NGPULayers,
		kvTypeK:         kvName(cfg.Context.TypeK),
		kvTypeV:         kvName(cfg.Context.TypeV),
		flashAttn:       cfg.Context.FlashAttn,
		loadMTP:         cfg.Model.LoadMTP,
		threads:         cfg.Context.NThreads,
		outputsMax:      cfg.Context.NOutputsMax,
		samplers:        make([]*Sampler, nSeq),
		ckpt:            make([][]byte, nSeq),
		custom:          make([]bool, nSeq),
		defaultSampling: cfg.Sampling,
	}

	// A reasoning block can only be primed away on a model that actually has one. Tokenizing
	// "<think>" to exactly one token proves it is a real special token rather than five
	// ordinary characters, and priming a model without it would push stray text into every
	// prompt.
	if toks, err := vocab.Tokenize("<think>", false, true); err == nil && len(toks) == 1 {
		r.thinkPrime = DefaultNoThinkPrime
	}

	for i := range r.samplers {
		s, err := NewSampler(cfg.Sampling, nVocab, vocab)
		if err != nil {
			r.Close()
			return nil, fmt.Errorf("sampler %d: %w", i, err)
		}
		r.samplers[i] = s
	}

	if cfg.MMProjPath != "" {
		v, err := OpenVision(m, VisionParams{
			MMProjPath: cfg.MMProjPath,
			UseGPU:     cfg.VisionGPU,
			NThreads:   int(cfg.Context.NThreads),
		})
		if err != nil {
			r.Close()
			return nil, fmt.Errorf("vision: %w", err)
		}
		r.vision = v
	}

	// A missing chat template is not fatal: the model can still serve raw completions.
	// It only means chat requests for it must be refused rather than rendered wrongly.
	if tmpl, err := m.ChatTemplate(); err == nil {
		r.chatTmpl = tmpl
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
	if r.vision != nil {
		r.vision.Free()
		r.vision = nil
	}
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
		if r.ownsModel {
			r.model.Free()
		}
		r.model = nil
	}
}

// Model exposes the loaded weights so a caller can build further runners over them.
func (r *Runner) Model() *Model { return r.model }

// HasVision reports whether this model can accept images.
func (r *Runner) HasVision() bool { return r.vision != nil }

// PrefillMedia encodes media with the prompt into seq's KV and returns the position
// generation continues from. The scheduler then decodes that sequence normally.
func (r *Runner) PrefillMedia(seq engine.SeqID, nPast engine.Pos, prompt string,
	media [][]byte, logitsLast bool) (engine.Pos, error) {
	if r.vision == nil {
		return 0, ErrNoProjector
	}
	p, err := r.vision.Prefill(r.ctx, SeqID(seq), Pos(nPast), prompt, media,
		r.batch.Cap(), logitsLast)
	return engine.Pos(p), err
}

// MediaMarker is the placeholder a prompt must contain where media belongs.
func (r *Runner) MediaMarker() string { return Marker() }

// Perf returns libllama's accounting for this runner's context.
func (r *Runner) Perf() Perf { return r.ctx.Perf() }

// PerfReset clears it, so warmup can be excluded from a measurement.
func (r *Runner) PerfReset() { r.ctx.PerfReset() }

// LibraryPerf reports the library's prefill accounting. Its decode counter is not reported:
// it increments only for single-token batches, so it reads near zero for a batching engine.
func (r *Runner) LibraryPerf(_ uint64) bench.LibraryPerf {
	p := r.ctx.Perf()
	return bench.LibraryPerf{
		PromptTokens:    p.PromptTokens,
		PromptTokPerSec: p.PromptTokensPerSec(),
		GraphReuse:      p.GraphReuse,
	}
}

// ResetLibraryPerf clears the library counters.
func (r *Runner) ResetLibraryPerf() { r.ctx.PerfReset() }

// PlacementInfo describes where this runner's weights went and how the context is configured.
//
// The key field is OnGPU. A device existing does not mean the model reached it: a manifest can
// ask for zero offload, or offload can fail, and the server then reports an accelerator while
// doing the work on CPU. That presents as a mysteriously slow GPU rather than an unused one.
type PlacementInfo struct {
	GPULayersRequested int32
	LayersTotal        int32
	OnGPU              bool
	ContextTotal       uint32
	ContextPerSeq      uint32
	BatchSize          uint32
	KVTypeK            string
	KVTypeV            string
	FlashAttn          bool
	MTPLoaded          bool
}

// PlacementInfo reports how this model was loaded.
func (r *Runner) PlacementInfo() PlacementInfo {
	sh := r.model.Shape()
	return PlacementInfo{
		GPULayersRequested: r.gpuLayers,
		LayersTotal:        int32(sh.Layers),
		// Any non-zero offload request means weights were placed on a device; zero means
		// the model is entirely on CPU whatever hardware is present.
		OnGPU:         r.gpuLayers != 0,
		ContextTotal:  r.ctx.NCtx(),
		ContextPerSeq: r.ctx.NCtxSeq(),
		BatchSize:     uint32(r.batch.Cap()),
		KVTypeK:       r.kvTypeK,
		KVTypeV:       r.kvTypeV,
		FlashAttn:     r.flashAttn,
		MTPLoaded:     r.loadMTP,
	}
}

// Summary describes the loaded model, including what it declares about MTP.
func (r *Runner) Summary() string { return r.model.Summary() }

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
		// Wait here rather than letting the first logit read do it, so the forward pass is
		// timed as the forward pass. See Context.Synchronize.
		r.ctx.Synchronize()
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

// SetSampling installs a per-request sampler chain for one sequence.
//
// Chains are rebuilt rather than mutated because llama.cpp's chain is a fixed pipeline once
// constructed. The cost is a handful of small allocations per request, which is negligible
// beside a forward pass — and the alternative, sharing one chain, would let concurrent
// streams read each other's penalty windows.
func (r *Runner) SetSampling(seq engine.SeqID, p *engine.SamplingParams) error {
	i := int(seq)
	if i < 0 || i >= len(r.samplers) {
		return fmt.Errorf("llama: sequence %d out of range (have %d)", seq, len(r.samplers))
	}

	if p.IsZero() {
		if !r.custom[i] {
			return nil // already on the default chain
		}
		s, err := NewSampler(r.defaultSampling, r.nVocab, r.vocab)
		if err != nil {
			return err
		}
		r.samplers[i].Free()
		r.samplers[i] = s
		r.custom[i] = false
		return nil
	}

	merged := r.defaultSampling
	if p.Temperature != nil {
		merged.Temperature = *p.Temperature
	}
	if p.TopK != nil {
		merged.TopK = *p.TopK
	}
	if p.TopP != nil {
		merged.TopP = *p.TopP
	}
	if p.MinP != nil {
		merged.MinP = *p.MinP
	}
	if p.RepeatLastN != nil {
		merged.RepeatLastN = *p.RepeatLastN
	}
	if p.RepeatPenalty != nil {
		merged.RepeatPenalty = *p.RepeatPenalty
	}
	if p.Grammar != nil {
		merged.Grammar = *p.Grammar
	}
	if p.GrammarRoot != nil {
		merged.GrammarRoot = *p.GrammarRoot
	}
	if p.FreqPenalty != nil {
		merged.FreqPenalty = *p.FreqPenalty
	}
	if p.PresencePenalty != nil {
		merged.PresencePenalty = *p.PresencePenalty
	}
	if p.Seed != nil {
		merged.Seed = *p.Seed
	}

	s, err := NewSampler(merged, r.nVocab, r.vocab)
	if err != nil {
		return err
	}
	r.samplers[i].Free()
	r.samplers[i] = s
	r.custom[i] = true
	return nil
}

// TrimSeq drops a sequence's cells from a position onward, keeping the prefix. Used to
// discard rejected speculative tokens without disturbing what was accepted.
// TrimSeq takes back the tail of a sequence from `from` onward.
//
// The result is not decoration: on an architecture that cannot rewind, the removal does
// nothing and reports so. Discarding that leaves the engine believing it rewound while the
// cache still holds the rejected tokens, and the next batch is rejected for inconsistent
// positions — a failure that names neither the rewind nor the speculation that needed it.
func (r *Runner) TrimSeq(seq engine.SeqID, from engine.Pos) bool {
	return r.ctx.SeqRm(SeqID(seq), Pos(from), -1)
}

// CanSeqRm reports how much of a sequence this runner can take back.
func (r *Runner) CanSeqRm() SeqRmSupport { return r.ctx.CanSeqRm() }

// Checkpoint snapshots the part of a sequence's state that a positional trim cannot rewind.
//
// Only the partial state is copied — the recurrent and sliding-window caches. The attention
// cache is far larger and is rewound by trimming instead, so copying it here would make a
// per-step snapshot cost more than the speculation it enables.
// checkpointFlags keep the snapshot in device memory.
//
// Without OnDevice every checkpoint copies state to host and every rollback copies it back —
// twice across the bus per decode step, per sequence, for state that never leaves the card
// otherwise. Speculation takes a checkpoint on every step it drafts, so that traffic is paid
// continuously rather than occasionally.
//
// One snapshot per sequence may be live at a time under this flag, which is exactly how it
// is used: taken before a step, consumed by that step's rollback.
const checkpointFlags = StateSeqPartialOnly | StateSeqOnDevice

func (r *Runner) Checkpoint(seq engine.SeqID) error {
	if int(seq) < 0 || int(seq) >= len(r.ckpt) {
		return fmt.Errorf("llama: sequence %d out of range for checkpointing", seq)
	}
	n := r.ctx.SeqStateSize(SeqID(seq), checkpointFlags)
	if !r.ckptLogged {
		r.ckptLogged = true
		log.Printf("engine: speculative checkpoint is %d bytes per step per sequence", n)
	}
	if n == 0 {
		// Nothing that needs carrying back. Recording an empty checkpoint keeps Rollback
		// honest about whether one was ever taken.
		r.ckpt[seq] = r.ckpt[seq][:0]
		return nil
	}
	if cap(r.ckpt[seq]) < n {
		r.ckpt[seq] = make([]byte, n)
	}
	r.ckpt[seq] = r.ckpt[seq][:n]
	if got := r.ctx.SeqStateSave(SeqID(seq), r.ckpt[seq], checkpointFlags); got == 0 {
		r.ckpt[seq] = r.ckpt[seq][:0]
		return fmt.Errorf("llama: could not snapshot sequence %d (%d bytes expected)", seq, n)
	}
	return nil
}

// Rollback restores the last checkpoint and trims the attention cache back to `to`.
//
// Both halves are needed and in this order: the restore carries back state that has no
// position, and the trim removes the positions written past the accepted point. Doing either
// alone leaves the sequence inconsistent in a way the next decode refuses.
func (r *Runner) Rollback(seq engine.SeqID, to engine.Pos) error {
	if int(seq) < 0 || int(seq) >= len(r.ckpt) {
		return fmt.Errorf("llama: sequence %d out of range for rollback", seq)
	}
	if buf := r.ckpt[seq]; len(buf) > 0 {
		if got := r.ctx.SeqStateLoad(SeqID(seq), buf, checkpointFlags); got == 0 {
			return fmt.Errorf("llama: sequence %d rejected its checkpoint", seq)
		}
	}
	r.ctx.SeqRm(SeqID(seq), Pos(to), -1)
	return nil
}

// DropCheckpoint releases the snapshot held for a finished sequence.
func (r *Runner) DropCheckpoint(seq engine.SeqID) {
	if int(seq) >= 0 && int(seq) < len(r.ckpt) {
		r.ckpt[seq] = r.ckpt[seq][:0]
	}
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

// Threads is the thread count this runner's context was created with.
func (r *Runner) Threads() int32 { return r.threads }

// OutputsMax is the logit-buffer bound this runner's context was created with. Zero means the
// library chose it from the batch size.
func (r *Runner) OutputsMax() uint32 { return r.outputsMax }
