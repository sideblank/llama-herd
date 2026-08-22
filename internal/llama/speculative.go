// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

/*
#cgo LDFLAGS: -llhspec
#include <stdlib.h>
#include "lhspec.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/sideblank/llama-herd/internal/engine"
)

// SpecTypesForModel reports which speculative types a model file supports, without loading
// the weights.
//
// This is the cheap gate for a build pipeline: a quantization that dropped its prediction
// head returns nothing here, and that answer costs a header read rather than a model load.
func SpecTypesForModel(path string) ([]string, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	buf := make([]byte, 256)
	n := C.lhspec_types_for_model(cpath, (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)))
	if n < 0 {
		return nil, fmt.Errorf("llama: could not read speculative types from %q", path)
	}
	if int(n) >= len(buf) {
		buf = make([]byte, int(n)+1)
		if n = C.lhspec_types_for_model(cpath, (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf))); n < 0 {
			return nil, fmt.Errorf("llama: could not read speculative types from %q", path)
		}
	}
	s := C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
	if s == "" {
		return nil, nil
	}
	return splitCSV(s), nil
}

func splitCSV(s string) []string {
	var out []string
	cur := ""
	for _, r := range s {
		if r == ',' || r == ' ' {
			if cur != "" {
				out = append(out, cur)
				cur = ""
			}
			continue
		}
		cur += string(r)
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}

// ErrNoSpeculation means the model supports no speculative type, so there is no head to drive.
var ErrNoSpeculation = errors.New("llama: model supports no speculative decoding")

// Speculative drafts tokens using the model's own prediction head.
//
// NOT SAFE FOR CONCURRENT USE. Like the rest of the backend it belongs to the decode-loop
// goroutine, which is also the only place it can correctly observe the target's state.
type Speculative struct {
	c      unsafe.Pointer
	runner *Runner
	maxN   int
	buf    []int32
}

var (
	_ engine.Drafter               = (*Speculative)(nil)
	_ engine.Seeder                = (*Speculative)(nil)
	_ engine.OutputAtEveryPosition = (*Speculative)(nil)
	_ engine.BatchObserver         = (*Speculative)(nil)
)

// OpenSpeculative creates a driver over a loaded runner.
//
// types selects which mechanisms to use, comma separated; empty accepts whatever the model
// supports. maxDraft bounds tokens proposed per step.
func OpenSpeculative(r *Runner, types string, maxDraft int) (*Speculative, error) {
	if maxDraft < 1 {
		maxDraft = 4
	}
	ctypes := C.CString(types)
	defer C.free(unsafe.Pointer(ctypes))

	// The draft context allocates its own KV cache and does not read the target's
	// configuration, so every field it shares with the target is passed explicitly. Letting
	// any of them default asks for the model's full trained context at f16 — which fails
	// allocation on a card the target fits, and reports that failure as "no speculation
	// available", the same thing a stripped head reports.
	ck, cv := C.CString(r.kvTypeK), C.CString(r.kvTypeV)
	defer C.free(unsafe.Pointer(ck))
	defer C.free(unsafe.Pointer(cv))
	flash := C.int32_t(0)
	if r.flashAttn {
		flash = 1
	}

	h := C.lhspec_init(unsafe.Pointer(r.model.c), unsafe.Pointer(r.ctx.c), ctypes,
		C.int32_t(r.NSeqMax()), C.int32_t(maxDraft),
		C.int32_t(r.NCtx()), ck, cv, flash)
	if h == nil {
		return nil, ErrNoSpeculation
	}
	return &Speculative{c: h, runner: r, maxN: maxDraft, buf: make([]int32, maxDraft)}, nil
}

// Close releases the driver.
func (s *Speculative) Close() {
	if s == nil || s.c == nil {
		return
	}
	C.lhspec_free(s.c)
	s.c = nil
}

// MaxDraft bounds proposals per step.
func (s *Speculative) MaxDraft() int { return s.maxN }

// Seed starts a generation, giving the prompt it continues from.
func (s *Speculative) Seed(seq engine.SeqID, tokens []engine.Token) {
	if s.c == nil {
		return
	}
	if len(tokens) == 0 {
		C.lhspec_begin(s.c, C.int32_t(seq), nil, 0)
		return
	}
	buf := make([]int32, len(tokens))
	for i, t := range tokens {
		buf[i] = int32(t)
	}
	C.lhspec_begin(s.c, C.int32_t(seq), (*C.int32_t)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)))
}

// ObserveDecode shows the driver the batch the target just processed.
//
// Required after every pass, not only before drafting: a head that predicts from hidden
// states reads them here, and skipping a pass leaves it working from stale state — which
// shows up as drafts quietly ceasing to be accepted rather than as an error.
func (s *Speculative) ObserveDecode() error {
	if s.c == nil {
		return nil
	}
	if rc := C.lhspec_process(s.c, unsafe.Pointer(&s.runner.batch.c)); rc < 0 {
		return fmt.Errorf("llama: speculative process failed (%d)", int32(rc))
	}
	return nil
}

// Draft proposes continuation tokens for one sequence.
func (s *Speculative) Draft(seq engine.SeqID, last engine.Token, pos engine.Pos, n int) ([]engine.Token, error) {
	if s.c == nil {
		return nil, nil
	}
	if n > s.maxN {
		n = s.maxN
	}
	if n < 1 {
		return nil, nil
	}

	got := C.lhspec_draft(s.c, C.int32_t(seq), C.int32_t(pos), C.int32_t(last), C.int32_t(n),
		(*C.int32_t)(unsafe.Pointer(&s.buf[0])), C.int32_t(len(s.buf)))
	if got < 0 {
		// Declining to draft is not a failure worth ending a request over; the engine
		// falls back to ordinary decoding.
		return nil, nil
	}
	out := make([]engine.Token, int(got))
	for i := range out {
		out[i] = engine.Token(s.buf[i])
	}
	return out, nil
}

// Accept reports how many proposals the target kept.
func (s *Speculative) Accept(seq engine.SeqID, accepted int, _ engine.Token) error {
	if s.c == nil {
		return nil
	}
	C.lhspec_accept(s.c, C.int32_t(seq), C.int32_t(accepted))
	return nil
}

// OutputAtEveryPosition reports that this drafter reads the target's hidden states.
//
// The head consumes the state of each position it predicts from, and those exist only where
// the target was asked for output. Requesting output only on the last prompt token leaves
// the head with nothing to read: it never fills its cache, never drafts, and reports no
// error — the same picture a model with no head presents.
func (s *Speculative) OutputAtEveryPosition() bool { return true }

// Release is a no-op: the driver keeps its own per-sequence state and resets it on the next
// Seed, so there is nothing to discard here.
func (s *Speculative) Release(_ engine.SeqID) {}
