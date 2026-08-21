// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

/*
#cgo LDFLAGS: -lmtmd
#include <stdlib.h>
#include <string.h>
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// Vision drives a model's multimodal projector, turning an image plus a prompt into tokens
// already resident in a sequence's KV cache.
//
// The shape is worth stating because it decides how the scheduler treats a vision request:
// the image is encoded and prefilled into the sequence directly, and generation then
// proceeds through the ordinary decode loop from the returned position. There is no separate
// vision decode path — only a different way of filling the prompt.
//
// NOT SAFE FOR CONCURRENT USE. Like Runner, it belongs to the decode-loop goroutine.
type Vision struct {
	c     *C.mtmd_context
	model *Model
}

// ErrNoProjector means no multimodal projector was configured, so the model is text-only.
var ErrNoProjector = errors.New("llama: model has no multimodal projector")

// VisionParams configures the projector.
type VisionParams struct {
	// MMProjPath is the projector file that accompanies a vision model.
	MMProjPath string
	// UseGPU offloads the vision encoder. Encoding an image on CPU is slow enough to
	// dominate time-to-first-token.
	UseGPU bool
	// NThreads is used when the encoder runs on CPU.
	NThreads int
}

// OpenVision initialises a multimodal context over an already-loaded model.
func OpenVision(m *Model, p VisionParams) (*Vision, error) {
	if p.MMProjPath == "" {
		return nil, ErrNoProjector
	}

	cpath := C.CString(p.MMProjPath)
	defer C.free(unsafe.Pointer(cpath))

	cp := C.mtmd_context_params_default()
	cp.use_gpu = C.bool(p.UseGPU)
	if p.NThreads > 0 {
		cp.n_threads = C.int(p.NThreads)
	}
	// Warmup runs an encode pass at init so the first real request does not pay for
	// lazily allocated buffers, which would otherwise land in the first measurement.
	cp.warmup = C.bool(true)

	ctx := C.mtmd_init_from_file(cpath, m.c, cp)
	if ctx == nil {
		return nil, fmt.Errorf("llama: could not load projector %q", p.MMProjPath)
	}
	return &Vision{c: ctx, model: m}, nil
}

// Free releases the multimodal context.
func (v *Vision) Free() {
	if v == nil || v.c == nil {
		return
	}
	C.mtmd_free(v.c)
	v.c = nil
}

// Marker is the placeholder that must appear in a prompt where the media belongs. A prompt
// without it produces a chunk list containing no media, so the image is silently ignored
// and the model answers about nothing.
func Marker() string { return C.GoString(C.mtmd_default_marker()) }

// Prefill encodes media alongside the prompt and writes the result into seq's KV cache,
// beginning at nPast. It returns the position generation should continue from.
//
// prompt must contain Marker() once per item in media. logitsLast leaves logits on the
// final token so the caller can sample immediately without another decode.
func (v *Vision) Prefill(ctx *Context, seq SeqID, nPast Pos, prompt string,
	media [][]byte, nBatch int32, logitsLast bool) (Pos, error) {

	if v == nil || v.c == nil {
		return 0, ErrNoProjector
	}
	if len(media) == 0 {
		return 0, errors.New("llama: no media supplied")
	}

	bitmaps := make([]*C.mtmd_bitmap, 0, len(media))
	defer func() {
		for _, b := range bitmaps {
			C.mtmd_bitmap_free(b)
		}
	}()

	for i, buf := range media {
		if len(buf) == 0 {
			return 0, fmt.Errorf("llama: media %d is empty", i)
		}
		w := C.mtmd_helper_bitmap_init_from_buf(v.c,
			(*C.uchar)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)), C.bool(false))
		if w.bitmap == nil {
			return 0, fmt.Errorf("llama: could not decode media %d (%d bytes)", i, len(buf))
		}
		bitmaps = append(bitmaps, w.bitmap)
	}

	cprompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cprompt))

	text := C.struct_mtmd_input_text{
		text:          cprompt,
		text_len:      C.size_t(len(prompt)),
		add_special:   C.bool(true),
		parse_special: C.bool(true),
	}

	chunks := C.mtmd_input_chunks_init()
	if chunks == nil {
		return 0, errors.New("llama: could not allocate input chunks")
	}
	defer C.mtmd_input_chunks_free(chunks)

	if rc := C.mtmd_tokenize(v.c, chunks, &text,
		(**C.mtmd_bitmap)(unsafe.Pointer(&bitmaps[0])), C.size_t(len(bitmaps))); rc != 0 {
		return 0, fmt.Errorf("llama: tokenize with media failed (%d) — is %q present in the prompt?",
			int32(rc), Marker())
	}

	var newPast C.llama_pos
	if rc := C.mtmd_helper_eval_chunks(v.c, ctx.c, chunks,
		C.llama_pos(nPast), C.llama_seq_id(seq), C.int32_t(nBatch),
		C.bool(logitsLast), &newPast); rc != 0 {
		return 0, fmt.Errorf("llama: evaluating media chunks failed (%d)", int32(rc))
	}

	return Pos(newPast), nil
}
