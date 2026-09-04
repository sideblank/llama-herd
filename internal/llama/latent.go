// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

/*
#include <stdlib.h>
#include "llama.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// EmbeddingsIth returns the hidden state for the i'th output position of the last Decode.
//
// This is the FINAL LAYER OUTPUT — the state that feeds the LM head — not an input embedding.
// The distinction matters: passing one of these to an embedding batch puts a vector in a space
// the model does not expect. See EmbdBatch.
//
// Requires the context to have been created with Embeddings set, and the position to have been
// staged with wantLogits. Returns nil when the library has nothing for that index, which is a
// configuration error rather than an empty result — check it.
func (ctx *Context) EmbeddingsIth(i int32, nEmbd int32) []float32 {
	p := C.llama_get_embeddings_ith(ctx.c, C.int32_t(i))
	if p == nil {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(p)), int(nEmbd))
}

// EmbeddingsSeq returns the pooled embedding for a sequence, when the context was created with a
// pooling type. Returns nil under LLAMA_POOLING_TYPE_NONE, which is the generative default.
func (ctx *Context) EmbeddingsSeq(seq SeqID, nEmbd int32) []float32 {
	p := C.llama_get_embeddings_seq(ctx.c, C.llama_seq_id(seq))
	if p == nil {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(p)), int(nEmbd))
}

// EmbdBatch stages input embeddings instead of token ids.
//
// A batch is one or the other, decided at allocation: `llama_batch_init` allocates `embd` when its
// second argument is non-zero and `token` otherwise. There is no mixing, so this is a separate
// type rather than a mode on Batch — a batch that silently ignored staged tokens because it was
// allocated for embeddings would be a bad afternoon.
//
// What this enables: seeding a stream from vectors rather than text. Vectors are consumed at
// layer 0, in the same place a token's embedding lookup would have deposited them.
//
// ⚠️ It is NOT a KV-cache prefix. KV entries are per-layer — 2 x n_layers x d_head vectors per
// position — and cannot be produced from a single final-layer state. Layer-0 injection is the only
// path the library exposes.
type EmbdBatch struct {
	c       C.struct_llama_batch
	cap     int32
	nEmbd   int32
	nSeqMax int32
}

// NewEmbdBatch allocates a batch holding up to nTokens vectors of nEmbd floats each.
func NewEmbdBatch(nTokens, nEmbd, nSeqMax int32) (*EmbdBatch, error) {
	if nTokens < 1 || nEmbd < 1 || nSeqMax < 1 {
		return nil, fmt.Errorf("llama: embd batch needs positive dimensions, got "+
			"tokens=%d embd=%d seqs=%d", nTokens, nEmbd, nSeqMax)
	}
	return &EmbdBatch{
		c:       C.llama_batch_init(C.int32_t(nTokens), C.int32_t(nEmbd), C.int32_t(nSeqMax)),
		cap:     nTokens,
		nEmbd:   nEmbd,
		nSeqMax: nSeqMax,
	}, nil
}

// Free releases the batch's C arrays.
func (b *EmbdBatch) Free() {
	if b == nil || b.cap == 0 {
		return
	}
	C.llama_batch_free(b.c)
	b.cap = 0
}

// Len is how many vectors are staged; Cap is the most it can hold.
func (b *EmbdBatch) Len() int32 { return int32(b.c.n_tokens) }
func (b *EmbdBatch) Cap() int32 { return b.cap }

// Clear empties the batch without freeing it.
func (b *EmbdBatch) Clear() { b.c.n_tokens = 0 }

// Add stages one vector at the given position.
//
// The vector must be exactly nEmbd long. A short one would leave the tail of that row holding
// whatever was in the heap — llama_batch_init leaves every member uninitialised — and the model
// would consume it as signal.
func (b *EmbdBatch) Add(vec []float32, pos Pos, seqs []SeqID, wantLogits bool) error {
	i := int32(b.c.n_tokens)
	if i >= b.cap {
		return ErrBatchFull
	}
	if int32(len(vec)) != b.nEmbd {
		return fmt.Errorf("llama: vector has %d elements, batch expects exactly %d — a short "+
			"vector leaves uninitialised heap in the row and the model reads it as signal",
			len(vec), b.nEmbd)
	}
	if len(seqs) == 0 {
		return errors.New("llama: a batch entry needs at least one sequence id")
	}
	if int32(len(seqs)) > b.nSeqMax {
		return fmt.Errorf("llama: %d sequence ids exceeds the batch's n_seq_max of %d",
			len(seqs), b.nSeqMax)
	}

	row := unsafe.Slice((*float32)(unsafe.Pointer(b.c.embd)), int(b.cap)*int(b.nEmbd))
	copy(row[int(i)*int(b.nEmbd):(int(i)+1)*int(b.nEmbd)], vec)

	unsafe.Slice(b.c.pos, b.cap)[i] = C.llama_pos(pos)
	unsafe.Slice(b.c.n_seq_id, b.cap)[i] = C.int32_t(len(seqs))
	seqRow := unsafe.Slice(unsafe.Slice(b.c.seq_id, b.cap)[i], b.nSeqMax)
	for j, s := range seqs {
		seqRow[j] = C.llama_seq_id(s)
	}
	var l C.int8_t
	if wantLogits {
		l = 1
	}
	unsafe.Slice(b.c.logits, b.cap)[i] = l

	b.c.n_tokens++
	return nil
}

// DecodeEmbd runs a forward pass over staged vectors.
func (ctx *Context) DecodeEmbd(b *EmbdBatch) error {
	if b == nil || b.Len() == 0 {
		return errors.New("llama: nothing staged in the embedding batch")
	}
	switch rc := int(C.llama_decode(ctx.c, b.c)); {
	case rc == 0:
		ctx.Synchronize()
		return nil
	case rc == 1:
		return ErrNoKVSlot
	default:
		return fmt.Errorf("llama: decode of an embedding batch failed (%d)", rc)
	}
}

// NEmbd is the model's hidden width, which every vector staged into an EmbdBatch must match.
func (m *Model) NEmbd() int32 { return int32(C.llama_model_n_embd(m.c)) }
