// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package engine is the scheduler: it owns the slot table, builds each batch, and drives
// one decode loop per model.
//
// It contains no C. Everything model-facing sits behind Backend, which keeps the scheduling
// logic testable without a GPU — the part most worth testing, since its failure modes are
// concurrency and budget bugs rather than numerics.
package engine

import "errors"

type (
	Token int32
	Pos   int32
	SeqID int32
)

// ChatMessage is one turn of a conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// SamplingParams is a backend-neutral description of how to draw the next token.
//
// Pointer fields distinguish "not specified" from a deliberate zero, because zero is
// meaningful for most of them: a temperature of 0 means greedy, not "use the default".
type SamplingParams struct {
	Temperature     *float32
	TopK            *int32
	TopP            *float32
	MinP            *float32
	RepeatLastN     *int32
	RepeatPenalty   *float32
	FreqPenalty     *float32
	PresencePenalty *float32
	Seed            *uint32
}

// IsZero reports whether nothing was specified, in which case the model's configured
// defaults apply and no per-request chain need be built.
func (p *SamplingParams) IsZero() bool {
	return p == nil || (p.Temperature == nil && p.TopK == nil && p.TopP == nil &&
		p.MinP == nil && p.RepeatLastN == nil && p.RepeatPenalty == nil &&
		p.FreqPenalty == nil && p.PresencePenalty == nil && p.Seed == nil)
}

// MediaBackend is implemented by backends that accept images or audio.
//
// It is a separate interface rather than part of Backend because most models are text-only
// and should not have to stub a method they cannot perform. The scheduler checks for it and
// refuses media requests to a backend that lacks it.
type MediaBackend interface {
	// PrefillMedia encodes media together with the prompt directly into seq's KV cache,
	// starting at nPast, and returns the position generation continues from.
	//
	// This replaces token-by-token prefill for that request; it batches internally.
	// Afterwards the sequence decodes through the ordinary loop like any other.
	PrefillMedia(seq SeqID, nPast Pos, prompt string, media [][]byte, logitsLast bool) (Pos, error)

	// MediaMarker is the placeholder the prompt must contain where media belongs.
	MediaMarker() string
}

// Drafter proposes continuation tokens so several can be verified in one forward pass.
//
// The draft source is deliberately abstract. A small companion model, a trained
// multi-token-prediction head, or an n-gram cache all produce the same thing — candidate
// tokens — and the verification loop does not care which. Keeping the interface narrow means
// the loop is written once.
type Drafter interface {
	// Draft proposes up to n continuation tokens for seq, following last at pos.
	// Returning fewer, including none, is valid and simply means less speculation.
	Draft(seq SeqID, last Token, pos Pos, n int) ([]Token, error)

	// Accept reports how many proposals the target kept and which token it chose instead
	// at the first divergence, so the drafter can keep its own state aligned.
	Accept(seq SeqID, accepted int, corrected Token) error

	// Release discards any state held for a finished sequence.
	Release(seq SeqID)

	// MaxDraft bounds how many tokens will be proposed at once.
	MaxDraft() int
}

// BatchObserver is an optional Drafter capability: a drafter that predicts from the target's
// internal state must see every decode, not only the ones where a draft is wanted.
//
// A trained prediction head works from the hidden states the target produced, so it has to be
// shown each batch as it is decoded. Calling it only when drafting would leave it predicting
// from state several steps stale, and the drafts would be wrong for a reason no counter would
// explain — they would simply stop being accepted.
type BatchObserver interface {
	// ObserveDecode is called after each successful forward pass, before drafts are
	// requested for the next one.
	ObserveDecode() error
}

// OutputAtEveryPosition is an optional Drafter capability: a drafter that reads the target's
// hidden states needs those states at every prompt position, not only the last.
//
// Ordinary prefill asks the target for output on the final prompt token alone, because that
// is the only position whose logits are sampled. A trained prediction head consumes the
// hidden state of each position it is going to predict from, and those states exist only
// where output was requested. Without them the head has nothing to read and simply never
// drafts — its cache stays empty and no counter says why.
//
// It is not free: every requested position occupies a row in the target's output buffer, so
// this trades memory during prefill for the head being usable at all.
type OutputAtEveryPosition interface {
	// OutputAtEveryPosition reports whether prefill must request output for every token.
	OutputAtEveryPosition() bool
}

// Seeder is an optional Drafter capability: a drafter that predicts from context wants the
// prompt, not just the tokens generated after it.
//
// This is where most of the benefit is for a lookup-based drafter — an agent's prompt carries
// the file it is editing, the schema it is filling, the transcript it is continuing, and the
// output repeats large spans of it.
type Seeder interface {
	Seed(seq SeqID, tokens []Token)
}

// Renderer turns messages into the prompt string a specific model expects.
//
// Unlike Backend, this is called from request goroutines and must be safe for concurrent
// use. Implementations should capture whatever they need at load time rather than reaching
// into live context state.
type Renderer interface {
	RenderChat(msgs []ChatMessage) (string, error)
}

var (
	// ErrNoKVSlot means the KV cache could not fit the batch. Recoverable: free a
	// sequence and resubmit.
	ErrNoKVSlot = errors.New("engine: no KV cache slot for batch")
	// ErrBatchFull means the batch has no room for another entry this tick.
	ErrBatchFull = errors.New("engine: batch full")
)

// Backend is the model-facing surface the scheduler needs. One Backend instance corresponds
// to one model's resident weights and its single llama context.
//
// Every method is called from the decode-loop goroutine only. Implementations therefore need
// no internal locking, which is the whole reason the single-loop design was chosen.
type Backend interface {
	// NCtx is the total context across all sequences.
	NCtx() uint32
	// NCtxSeq is the context available to ONE sequence. This, not NCtx, is what a
	// single prompt has to fit inside: the total is shared across every slot, so
	// admitting against it accepts prompts that cannot actually run.
	NCtxSeq() uint32
	// NSeqMax is how many sequences the context can hold, i.e. the slot count.
	NSeqMax() uint32
	// BatchCap is the largest number of entries one Decode may carry.
	BatchCap() int32

	// Tokenize converts prompt text to tokens.
	Tokenize(text string, addSpecial bool) ([]Token, error)
	// Piece renders one token as the bytes it contributes. May be a partial UTF-8
	// sequence, so callers must buffer rather than assume validity.
	Piece(t Token) ([]byte, error)
	// EOS is the end-of-sequence token, or -1 if the model defines none.
	EOS() Token

	// BatchClear empties the staging batch for a new tick.
	BatchClear()
	// BatchAdd stages one token. wantLogits marks a position whose logits will be read
	// after Decode — true for the final prefill token and every decode token.
	BatchAdd(tok Token, pos Pos, seq SeqID, wantLogits bool) error
	// BatchLen is how many entries are staged.
	BatchLen() int32

	// Decode runs one forward pass over the staged batch.
	Decode() error

	// SampleAt draws the next token from the logits at batch index i, for sequence seq.
	// The sampler state is per-sequence, so seq selects which chain advances.
	SampleAt(seq SeqID, i int32) (Token, error)

	// SetSampling configures the sampler for one sequence before its first sample.
	// Passing nil restores the model's configured defaults.
	//
	// Sampler state is per-sequence, so this replaces that sequence's chain and must
	// not disturb any other stream's.
	SetSampling(seq SeqID, p *SamplingParams) error

	// TrimSeq drops a sequence's KV cells from position p onward, keeping everything
	// before it.
	//
	// Speculation needs this: drafted tokens are written into the cache to be verified,
	// and the rejected tail must be removed or the sequence continues from a state the
	// model never actually chose.
	// TrimSeq removes a sequence's cache from `from` onward, reporting whether it could.
	//
	// It returns false on architectures that cannot rewind — those carrying recurrent
	// state, where only a whole sequence can be dropped. Speculation depends on this
	// working, because a rejected draft is written into the cache to be checked and must
	// then be taken back. Ignoring the result leaves the cache and the engine's idea of
	// position disagreeing, which surfaces later as a rejected batch.
	TrimSeq(seq SeqID, from Pos) bool

	// FreeSeq drops a sequence's KV cells, returning its capacity to the pool.
	FreeSeq(seq SeqID)
}
