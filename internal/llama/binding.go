// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package llama is the binding to libllama. It is the only package in the tree that
// touches C: everything above it works in Go types.
//
// Build requirements. The header and library locations are not hard-coded; supply them
// through the standard cgo environment variables, e.g.
//
//	export CGO_CFLAGS="-I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include"
//	export CGO_LDFLAGS="-L/path/to/llama.cpp/build/bin"
//
// Threading contract. libllama's context is not safe for concurrent use. Exactly one
// goroutine — the engine's decode loop — may call Decode, LogitsIth, or the memory
// methods on a given Context. This package does not lock; the caller owns that discipline.
package llama

/*
// libllama depends on ggml, and modern GNU ld does not resolve indirect shared-library
// dependencies transitively — linking only -lllama fails with "DSO missing from command
// line". The CUDA and CPU backends are not listed: ggml loads those as plugins at run time.
#cgo LDFLAGS: -lllama -lggml -lggml-base
#include <stdlib.h>
#include <string.h>
#include "llama.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// Token, Pos and SeqID mirror the C types rather than aliasing int, so a caller cannot
// silently pass the wrong one.
type (
	Token int32
	Pos   int32
	SeqID int32
)

var (
	// ErrNoKVSlot is llama_decode's return of 1: the KV cache has no room for the batch.
	// It is recoverable — free a sequence and retry — and must not be treated as fatal.
	ErrNoKVSlot = errors.New("llama: no KV cache slot available for the batch")
	// ErrDecode is any other non-zero llama_decode return. The context is unusable.
	ErrDecode = errors.New("llama: decode failed")
)

// Backend initialises libllama's global state and registers the backend plugins. Call once
// per process before loading a model; call BackendFree at shutdown.
//
// The plugin registration is not optional — see LoadBackends. Without it a build with
// dynamically loaded backends finds no devices at all and falls back to CPU silently.
func Backend() {
	LoadBackends()
	C.llama_backend_init()
}

// BackendFree releases libllama's global state.
func BackendFree() { C.llama_backend_free() }

// SystemInfo reports the CPU features and backends the linked libllama was built with.
// Worth recording in a bug report: a throughput number means little without it.
func SystemInfo() string { return C.GoString(C.llama_print_system_info()) }

// ---------------------------------------------------------------- model

// ModelParams is the subset of llama_model_params this project sets. Anything not named
// here keeps libllama's default.
type ModelParams struct {
	// NGPULayers is how many layers to place in VRAM. Negative means all of them.
	NGPULayers int32
	// MainGPU is the device index that holds the whole model when SplitMode is
	// SplitNone. Ignored otherwise.
	MainGPU int32
	// SplitMode selects how a model is spread across several devices.
	SplitMode SplitMode
	// TensorSplit gives the proportion of the model each device receives, indexed by
	// device. Empty means split evenly. This is what makes a mixed host work: a 24 GB
	// card and a 12 GB card should not receive equal shares.
	TensorSplit []float32
	// LoadMode selects mmap, mlock, direct I/O, or a combination. It replaced the
	// separate boolean flags upstream.
	LoadMode LoadMode
	// VocabOnly loads metadata and the vocabulary without the weights. Useful for
	// inspecting a file cheaply.
	VocabOnly bool
	// LoadMTP loads the model's multi-token-prediction layers, if the file carries
	// them. Many third-party quantizations strip these tensors, in which case there is
	// nothing to load and speculative decoding through the MTP head is unavailable.
	LoadMTP bool
}

// GGMLType identifies a tensor element type, used here for KV cache precision.
type GGMLType int32

const (
	TypeF16  GGMLType = C.GGML_TYPE_F16
	TypeQ8_0 GGMLType = C.GGML_TYPE_Q8_0
	TypeQ5_1 GGMLType = C.GGML_TYPE_Q5_1
	TypeQ4_0 GGMLType = C.GGML_TYPE_Q4_0
)

// ParseGGMLType maps a manifest string to a type. An empty string means f16.
func ParseGGMLType(s string) (GGMLType, bool) {
	switch s {
	case "", "f16":
		return TypeF16, true
	case "q8_0":
		return TypeQ8_0, true
	case "q5_1":
		return TypeQ5_1, true
	case "q4_0":
		return TypeQ4_0, true
	default:
		return TypeF16, false
	}
}

// SplitMode selects how a model is distributed across devices.
type SplitMode int32

const (
	// SplitNone keeps the whole model on MainGPU.
	SplitNone SplitMode = C.LLAMA_SPLIT_MODE_NONE
	// SplitLayer divides layers and KV cache across devices. The usual choice for a
	// model too large for one card.
	SplitLayer SplitMode = C.LLAMA_SPLIT_MODE_LAYER
	// SplitRow divides layers and KV, using tensor parallelism where supported.
	SplitRow SplitMode = C.LLAMA_SPLIT_MODE_ROW
	// SplitTensor is full tensor parallelism.
	SplitTensor SplitMode = C.LLAMA_SPLIT_MODE_TENSOR
)

// LoadMode selects how weights are brought into memory.
type LoadMode int32

const (
	LoadModeAuto      LoadMode = C.LLAMA_LOAD_MODE_AUTO
	LoadModeNone      LoadMode = C.LLAMA_LOAD_MODE_NONE
	LoadModeMmap      LoadMode = C.LLAMA_LOAD_MODE_MMAP
	LoadModeMlock     LoadMode = C.LLAMA_LOAD_MODE_MLOCK
	LoadModeMmapMlock LoadMode = C.LLAMA_LOAD_MODE_MMAP_MLOCK
	LoadModeDirectIO  LoadMode = C.LLAMA_LOAD_MODE_DIRECT_IO
)

// DefaultModelParams returns libllama's defaults, so callers override rather than
// construct a params struct from zero and accidentally disable mmap.
func DefaultModelParams() ModelParams {
	c := C.llama_model_default_params()
	return ModelParams{
		NGPULayers: int32(c.n_gpu_layers),
		MainGPU:    int32(c.main_gpu),
		SplitMode:  SplitMode(c.split_mode),
		LoadMode:   LoadMode(c.load_mode),
		VocabOnly:  bool(c.vocab_only),
		LoadMTP:    bool(c.load_mtp),
	}
}

// c builds the C params. The returned free function releases the tensor-split array and
// must be called once the load has completed.
func (p ModelParams) c() (C.struct_llama_model_params, func()) {
	c := C.llama_model_default_params()
	c.n_gpu_layers = C.int32_t(p.NGPULayers)
	c.main_gpu = C.int32_t(p.MainGPU)
	c.split_mode = C.enum_llama_split_mode(p.SplitMode)
	c.load_mode = C.enum_llama_load_mode(p.LoadMode)
	c.vocab_only = C.bool(p.VocabOnly)
	c.load_mtp = C.bool(p.LoadMTP)

	split, free := cTensorSplit(p.TensorSplit)
	c.tensor_split = split
	return c, free
}

// Model is a set of weights resident on the device. One Model is shared by every stream
// and every Context built from it — that sharing is the point of this project.
type Model struct {
	c *C.struct_llama_model
}

// LoadModel reads a GGUF file and places its layers according to params.
func LoadModel(path string, params ModelParams) (*Model, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	cparams, freeSplit := params.c()
	defer freeSplit()

	m := C.llama_model_load_from_file(cpath, cparams)
	if m == nil {
		return nil, fmt.Errorf("llama: could not load model %q", path)
	}
	return &Model{c: m}, nil
}

// Free releases the weights. Every Context built from this Model must be freed first.
func (m *Model) Free() {
	if m == nil || m.c == nil {
		return
	}
	C.llama_model_free(m.c)
	m.c = nil
}

// Vocab returns the model's vocabulary, which outlives nothing: it is owned by the Model
// and is invalid once the Model is freed.
func (m *Model) Vocab() *Vocab { return &Vocab{c: C.llama_model_get_vocab(m.c)} }

// ---------------------------------------------------------------- context

// ContextParams is the subset of llama_context_params this project sets.
type ContextParams struct {
	// NCtx is the total context across all sequences. 0 takes the model's training value.
	NCtx uint32
	// NBatch is the largest batch that may be submitted to Decode in one call. It bounds
	// how much prefill and how many concurrent decode tokens can ride a single pass.
	NBatch uint32
	// NUBatch is the physical micro-batch size.
	NUBatch uint32
	// NSeqMax is the number of distinct sequences, i.e. the number of concurrent streams
	// this context can host.
	NSeqMax uint32

	NThreads      int32
	NThreadsBatch int32

	// Embeddings makes the context produce embeddings alongside logits.
	Embeddings bool
	// OffloadKQV places the KV cache and attention ops on the GPU.
	OffloadKQV bool
	// TypeK and TypeV are the KV cache element types. Keys tolerate quantization less
	// well than values, so an asymmetric pair — keys at q8, values at q4 — is often a
	// better trade than one level for both.
	//
	// Any quantized type requires FlashAttn. An asymmetric pair additionally requires a
	// build with all flash-attention quant kernels compiled; the default build carries
	// only matched pairs.
	TypeK GGMLType
	TypeV GGMLType

	// FlashAttn enables flash attention, which quantized KV requires.
	FlashAttn bool

	// KVUnified selects one attention buffer shared across sequences instead of a
	// per-sequence one.
	//
	// The right answer depends on the workload, not on preference. Independent requests
	// that share nothing are better served per-sequence. Requests fanned out from a
	// common prompt — the same system message and context issued many ways — share a
	// large prefix, and a unified buffer exploits that. Upstream also warns that
	// disabling it with several sequences can hurt in some cases, so this is worth
	// measuring per deployment rather than assuming.
	KVUnified bool
	// CtxType selects the context kind. Use CtxTypeMTP for a context that drives the
	// model's multi-token-prediction head, which requires the weights to have been
	// loaded with LoadMTP.
	CtxType ContextType
}

// ContextType distinguishes an ordinary decode context from a multi-token-prediction one.
type ContextType int32

const (
	CtxTypeDefault ContextType = C.LLAMA_CONTEXT_TYPE_DEFAULT
	CtxTypeMTP     ContextType = C.LLAMA_CONTEXT_TYPE_MTP
)

// DefaultContextParams returns libllama's defaults.
func DefaultContextParams() ContextParams {
	c := C.llama_context_default_params()
	return ContextParams{
		NCtx:          uint32(c.n_ctx),
		NBatch:        uint32(c.n_batch),
		NUBatch:       uint32(c.n_ubatch),
		NSeqMax:       uint32(c.n_seq_max),
		NThreads:      int32(c.n_threads),
		NThreadsBatch: int32(c.n_threads_batch),
		Embeddings:    bool(c.embeddings),
		OffloadKQV:    bool(c.offload_kqv),
		TypeK:         GGMLType(c.type_k),
		TypeV:         GGMLType(c.type_v),
		FlashAttn:     c.flash_attn_type != C.LLAMA_FLASH_ATTN_TYPE_DISABLED,
		KVUnified:     bool(c.kv_unified),
		CtxType:       ContextType(c.ctx_type),
	}
}

func (p ContextParams) c() C.struct_llama_context_params {
	c := C.llama_context_default_params()
	c.n_ctx = C.uint32_t(p.NCtx)
	c.n_batch = C.uint32_t(p.NBatch)
	c.n_ubatch = C.uint32_t(p.NUBatch)
	c.n_seq_max = C.uint32_t(p.NSeqMax)
	c.n_threads = C.int32_t(p.NThreads)
	c.n_threads_batch = C.int32_t(p.NThreadsBatch)
	c.embeddings = C.bool(p.Embeddings)
	c.offload_kqv = C.bool(p.OffloadKQV)
	c.type_k = C.enum_ggml_type(p.TypeK)
	c.type_v = C.enum_ggml_type(p.TypeV)
	if p.FlashAttn {
		c.flash_attn_type = C.LLAMA_FLASH_ATTN_TYPE_ENABLED
	}
	c.kv_unified = C.bool(p.KVUnified)
	c.ctx_type = C.enum_llama_context_type(p.CtxType)
	return c
}

// Context holds the KV cache and compute state for a set of sequences over one Model.
//
// NOT SAFE FOR CONCURRENT USE. See the package comment.
type Context struct {
	c   *C.struct_llama_context
	mem C.llama_memory_t
}

// NewContext creates a context over m.
func NewContext(m *Model, params ContextParams) (*Context, error) {
	c := C.llama_init_from_model(m.c, params.c())
	if c == nil {
		return nil, errors.New("llama: could not create context")
	}
	return &Context{c: c, mem: C.llama_get_memory(c)}, nil
}

// Free releases the context and its KV cache.
func (ctx *Context) Free() {
	if ctx == nil || ctx.c == nil {
		return
	}
	C.llama_free(ctx.c)
	ctx.c = nil
}

// NCtx is the total context this Context was created with, shared across all sequences.
// It may differ from what was requested when 0 was passed.
func (ctx *Context) NCtx() uint32 { return uint32(C.llama_n_ctx(ctx.c)) }

// NCtxSeq is the context available to a single sequence. This is what one prompt plus its
// generation must fit inside, and it is what admission should be checked against.
func (ctx *Context) NCtxSeq() uint32 { return uint32(C.llama_n_ctx_seq(ctx.c)) }

// Decode runs one forward pass over batch.
//
// This is the call every stream's next token rides. A return of ErrNoKVSlot means the
// cache is full, not that anything is broken: evict a finished sequence with SeqRm and
// submit again.
func (ctx *Context) Decode(b *Batch) error {
	switch rc := int32(C.llama_decode(ctx.c, b.c)); {
	case rc == 0:
		return nil
	case rc == 1:
		return ErrNoKVSlot
	default:
		return fmt.Errorf("%w: llama_decode returned %d", ErrDecode, rc)
	}
}

// LogitsIth returns the logits for the i'th token of the last Decode, for the tokens whose
// logits flag was set. The slice aliases libllama's internal buffer: it is valid only until
// the next Decode, and must not be retained or mutated.
func (ctx *Context) LogitsIth(i int32, nVocab int32) []float32 {
	p := C.llama_get_logits_ith(ctx.c, C.int32_t(i))
	if p == nil {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(p)), int(nVocab))
}

// SeqRm drops a sequence's KV cells in [p0, p1). Passing -1 for p0 means the start and -1
// for p1 means the end, so SeqRm(id, -1, -1) frees a finished stream entirely.
func (ctx *Context) SeqRm(seq SeqID, p0, p1 Pos) bool {
	return bool(C.llama_memory_seq_rm(ctx.mem, C.llama_seq_id(seq), C.llama_pos(p0), C.llama_pos(p1)))
}

// ---------------------------------------------------------------- vocab

// Vocab is a model's token vocabulary. It is owned by the Model.
type Vocab struct {
	c *C.struct_llama_vocab
}

// NTokens is the vocabulary size, which is also the width of a logits row.
func (v *Vocab) NTokens() int32 { return int32(C.llama_vocab_n_tokens(v.c)) }

// BOS and EOS return the sentinel tokens, or -1 when the model defines none.
func (v *Vocab) BOS() Token { return Token(C.llama_vocab_bos(v.c)) }
func (v *Vocab) EOS() Token { return Token(C.llama_vocab_eos(v.c)) }

// Tokenize converts text to tokens. addSpecial adds BOS/EOS where the model expects them;
// parseSpecial makes control tokens in the text be recognised rather than escaped.
func (v *Vocab) Tokenize(text string, addSpecial, parseSpecial bool) ([]Token, error) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	clen := C.int32_t(len(text))

	// A negative return is minus the required length, so size the buffer from a probe
	// rather than guessing a bound and silently truncating long prompts.
	n := C.llama_tokenize(v.c, ctext, clen, nil, 0, C.bool(addSpecial), C.bool(parseSpecial))
	if n >= 0 {
		return nil, nil
	}
	need := int(-n)

	out := make([]Token, need)
	got := C.llama_tokenize(v.c, ctext, clen,
		(*C.llama_token)(unsafe.Pointer(&out[0])), C.int32_t(need),
		C.bool(addSpecial), C.bool(parseSpecial))
	if got < 0 {
		return nil, fmt.Errorf("llama: tokenize failed, wanted %d tokens", need)
	}
	return out[:int(got)], nil
}

// TokenToPiece renders one token as the bytes it contributes to the output. The result can
// be an incomplete UTF-8 sequence: a multi-byte rune may span several tokens, so callers
// streaming text must buffer rather than assume each piece is valid UTF-8.
func (v *Vocab) TokenToPiece(t Token, special bool) ([]byte, error) {
	buf := make([]byte, 32)
	n := C.llama_token_to_piece(v.c, C.llama_token(t),
		(*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)), 0, C.bool(special))
	if n < 0 {
		buf = make([]byte, int(-n))
		n = C.llama_token_to_piece(v.c, C.llama_token(t),
			(*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)), 0, C.bool(special))
		if n < 0 {
			return nil, fmt.Errorf("llama: token_to_piece failed for token %d", t)
		}
	}
	return buf[:int(n)], nil
}

// ---------------------------------------------------------------- batch

// Batch is one llama_decode payload. Tokens from many sequences share it: that is what
// makes a single decode pass serve every active stream.
//
// The C arrays are allocated once by llama_batch_init and reused, so a decode loop builds
// a batch per tick without allocating.
type Batch struct {
	c       C.struct_llama_batch
	cap     int32
	nSeqMax int32
}

// NewBatch allocates a batch able to hold nTokens entries, each addressable to at most
// nSeqMax sequences.
func NewBatch(nTokens, nSeqMax int32) *Batch {
	return &Batch{
		c:       C.llama_batch_init(C.int32_t(nTokens), 0, C.int32_t(nSeqMax)),
		cap:     nTokens,
		nSeqMax: nSeqMax,
	}
}

// Free releases the batch's C arrays.
func (b *Batch) Free() {
	if b == nil {
		return
	}
	C.llama_batch_free(b.c)
	b.cap = 0
}

// Len is how many tokens are currently staged.
func (b *Batch) Len() int32 { return int32(b.c.n_tokens) }

// Cap is the most tokens this batch can hold.
func (b *Batch) Cap() int32 { return b.cap }

// Clear empties the batch without freeing it, ready for the next tick.
func (b *Batch) Clear() { b.c.n_tokens = 0 }

// ErrBatchFull is returned by Add when the batch has no room. The caller decides whether
// to decode what it has and continue, or to defer the token.
var ErrBatchFull = errors.New("llama: batch is full")

// Add stages one token belonging to seqs. wantLogits marks it as a position whose logits
// will be read after Decode — set it for the last token of a prefill and for every decode
// token, and leave it false for interior prefill tokens so libllama can skip that work.
func (b *Batch) Add(tok Token, pos Pos, seqs []SeqID, wantLogits bool) error {
	i := int32(b.c.n_tokens)
	if i >= b.cap {
		return ErrBatchFull
	}
	if len(seqs) == 0 {
		return errors.New("llama: a batch entry needs at least one sequence id")
	}
	if int32(len(seqs)) > b.nSeqMax {
		return fmt.Errorf("llama: %d sequence ids exceeds the batch's n_seq_max of %d", len(seqs), b.nSeqMax)
	}

	unsafe.Slice(b.c.token, b.cap)[i] = C.llama_token(tok)
	unsafe.Slice(b.c.pos, b.cap)[i] = C.llama_pos(pos)
	unsafe.Slice(b.c.n_seq_id, b.cap)[i] = C.int32_t(len(seqs))

	row := unsafe.Slice(unsafe.Slice(b.c.seq_id, b.cap)[i], b.nSeqMax)
	for j, s := range seqs {
		row[j] = C.llama_seq_id(s)
	}

	var l C.int8_t
	if wantLogits {
		l = 1
	}
	unsafe.Slice(b.c.logits, b.cap)[i] = l

	b.c.n_tokens++
	return nil
}
