// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

/*
#include <stdlib.h>
#include "llama.h"
*/
import "C"

import (
	"fmt"
	"strconv"
	"strings"
	"unsafe"
)

// Meta reads one metadata key from the model file. Returns false when the key is absent.
func (m *Model) Meta(key string) (string, bool) {
	ckey := C.CString(key)
	defer C.free(unsafe.Pointer(ckey))

	buf := make([]byte, 256)
	n := C.llama_model_meta_val_str(m.c, ckey, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)))
	if n < 0 {
		return "", false
	}
	if int(n) >= len(buf) {
		buf = make([]byte, int(n)+1)
		n = C.llama_model_meta_val_str(m.c, ckey, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)))
		if n < 0 {
			return "", false
		}
	}
	return string(buf[:n]), true
}

// MetaAll returns every metadata key and value.
func (m *Model) MetaAll() map[string]string {
	n := int(C.llama_model_meta_count(m.c))
	out := make(map[string]string, n)
	kbuf := make([]byte, 256)
	vbuf := make([]byte, 1024)
	for i := 0; i < n; i++ {
		kn := C.llama_model_meta_key_by_index(m.c, C.int32_t(i),
			(*C.char)(unsafe.Pointer(&kbuf[0])), C.size_t(len(kbuf)))
		if kn < 0 {
			continue
		}
		vn := C.llama_model_meta_val_str_by_index(m.c, C.int32_t(i),
			(*C.char)(unsafe.Pointer(&vbuf[0])), C.size_t(len(vbuf)))
		if vn < 0 {
			continue
		}
		out[string(kbuf[:kn])] = string(vbuf[:vn])
	}
	return out
}

// Architecture is the model's architecture name, e.g. "qwen35".
func (m *Model) Architecture() string {
	if v, ok := m.Meta("general.architecture"); ok {
		return v
	}
	return "unknown"
}

// MTPInfo describes what a model file claims about multi-token prediction.
//
// This distinguishes two things that are easy to conflate and that model cards routinely
// blur: a file can *declare* MTP layers in its metadata while carrying none of the tensors,
// which is what most redistributed quantizations do. A declared count above zero is
// necessary but not sufficient — the tensors have to survive quantization too.
type MTPInfo struct {
	// DeclaredLayers is the count the file's metadata claims, or 0 if it claims none.
	DeclaredLayers int
	// MetaKey is the architecture-specific key the count came from.
	MetaKey string
}

// MTP reports the model's declared multi-token-prediction layers.
//
// The metadata key is architecture-prefixed, so it is derived from the architecture rather
// than hard-coded to any one model family.
func (m *Model) MTP() MTPInfo {
	arch := m.Architecture()
	for _, suffix := range []string{"nextn_predict_layers", "num_nextn_predict_layers", "mtp_layers"} {
		key := arch + "." + suffix
		if v, ok := m.Meta(key); ok {
			n, err := strconv.Atoi(strings.TrimSpace(v))
			if err == nil {
				return MTPInfo{DeclaredLayers: n, MetaKey: key}
			}
		}
	}
	return MTPInfo{}
}

// Summary is a one-line description for logs and reports.
func (m *Model) Summary() string {
	mtp := m.MTP()
	s := fmt.Sprintf("arch=%s layers=%d embd=%d ctx_train=%d",
		m.Architecture(),
		int(C.llama_model_n_layer(m.c)),
		int(C.llama_model_n_embd(m.c)),
		int(C.llama_model_n_ctx_train(m.c)))
	if mtp.DeclaredLayers > 0 {
		s += fmt.Sprintf(" mtp_declared=%d", mtp.DeclaredLayers)
	} else {
		s += " mtp_declared=0"
	}
	return s
}
