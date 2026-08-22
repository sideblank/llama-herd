// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

//go:build llamaabi

// This file exists only under the llamaabi tag, because it references llama.h enumerators that
// older revisions do not define. The point of the load-mode constants being literals is that
// this package builds against those revisions; the point of this file is that the literals are
// still checked against the header wherever it has them.
//
// Run with: go test -tags llamaabi ./internal/llama/

package llama

// #include "llama.h"
import "C"

// cLoadModes is the header's view of the enum, for comparison against ours.
var cLoadModes = map[string]int32{
	"auto":       int32(C.LLAMA_LOAD_MODE_AUTO),
	"none":       int32(C.LLAMA_LOAD_MODE_NONE),
	"mmap":       int32(C.LLAMA_LOAD_MODE_MMAP),
	"mlock":      int32(C.LLAMA_LOAD_MODE_MLOCK),
	"mmap_mlock": int32(C.LLAMA_LOAD_MODE_MMAP_MLOCK),
	"direct_io":  int32(C.LLAMA_LOAD_MODE_DIRECT_IO),
}
