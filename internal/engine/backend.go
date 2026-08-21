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

	// FreeSeq drops a sequence's KV cells, returning its capacity to the pool.
	FreeSeq(seq SeqID)
}
