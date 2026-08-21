// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

/*
#include "llama.h"
*/
import "C"

// Perf is libllama's own accounting for a context.
//
// Worth having alongside wall-clock measurement for two reasons. It separates prefill from
// decode at the library level, which is the ground truth the benchmark's own split should
// agree with — a large divergence means the harness is measuring its own overhead. And it
// counts decode *evaluations* separately from tokens produced, which is the only way to see
// speculative decoding working.
type Perf struct {
	// LoadMS is model load time.
	LoadMS float64
	// PromptMS and PromptTokens describe prefill.
	PromptMS     float64
	PromptTokens int32
	// EvalMS and EvalCount describe decode. EvalCount is the number of decode
	// evaluations, not the number of tokens produced.
	EvalMS    float64
	EvalCount int32
	// GraphReuse counts compute graphs reused rather than rebuilt.
	GraphReuse int32
}

// Perf returns the context's accounting.
func (ctx *Context) Perf() Perf {
	d := C.llama_perf_context(ctx.c)
	return Perf{
		LoadMS:       float64(d.t_load_ms),
		PromptMS:     float64(d.t_p_eval_ms),
		PromptTokens: int32(d.n_p_eval),
		EvalMS:       float64(d.t_eval_ms),
		EvalCount:    int32(d.n_eval),
		GraphReuse:   int32(d.n_reused),
	}
}

// PerfReset clears the counters, so a measurement can exclude warmup.
func (ctx *Context) PerfReset() { C.llama_perf_context_reset(ctx.c) }

// PromptTokensPerSec is prefill throughput as libllama measured it.
func (p Perf) PromptTokensPerSec() float64 {
	if p.PromptMS <= 0 {
		return 0
	}
	return float64(p.PromptTokens) / (p.PromptMS / 1000)
}

// EvalTokensPerSec is decode throughput as libllama measured it, counting evaluations.
func (p Perf) EvalTokensPerSec() float64 {
	if p.EvalMS <= 0 {
		return 0
	}
	return float64(p.EvalCount) / (p.EvalMS / 1000)
}

// TokensPerEval is produced tokens divided by decode evaluations, given the produced count.
//
// This is how speculative decoding shows up. Without it every evaluation yields exactly one
// token and the ratio is 1.0. With a multi-token-prediction head proposing and the target
// accepting, one evaluation yields more than one token and the ratio rises — so this is the
// acceptance rate, measurable without any dedicated API.
//
// A ratio at 1.0 on a model whose weights carry MTP layers means the head is loaded but
// contributing nothing, which is the failure worth catching: the VRAM is spent either way.
func (p Perf) TokensPerEval(produced uint64) float64 {
	if p.EvalCount <= 0 {
		return 0
	}
	return float64(produced) / float64(p.EvalCount)
}
