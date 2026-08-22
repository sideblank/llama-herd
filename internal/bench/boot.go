// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package bench

import (
	"context"
	"fmt"
	"time"

	"github.com/sideblank/llama-herd/internal/engine"
)

// Selftest is what this deployment actually delivers, measured once at startup through the
// engine that will serve the traffic.
//
// It measures the herd, not the model. A single-stream figure is the model's speed and can be
// looked up; what cannot be looked up is whether THIS deployment, on THIS card, with THIS
// stream count and context, reaches the throughput the arrangement is supposed to buy. That is
// the claim the design rests on — one resident copy of the weights serving several streams for
// close to the cost of one — and it is the thing that silently fails.
//
// It runs at the configured stream count for that reason. Measuring one stream would report a
// number that looks fine while the herd delivers nothing.
type Selftest struct {
	// Streams is how many ran concurrently: the arrangement being measured.
	Streams int `json:"streams"`
	// AggregateTokPerSec is every stream's tokens over the wall time they shared. This is the
	// number the deployment is worth.
	AggregateTokPerSec float64 `json:"aggregate_tok_per_sec"`
	// PerStreamTokPerSec is the aggregate divided by the stream count — what one caller sees.
	PerStreamTokPerSec float64 `json:"per_stream_tok_per_sec"`
	// TokensPerPass is how many tokens each forward pass carried. Near the stream count means
	// the herd is batching; near one means it is not, whatever the aggregate says.
	TokensPerPass float64 `json:"tokens_per_pass"`
	// PromptTokPerSec is prefill, reported separately because blending it into a decode rate
	// is how a throughput figure becomes uncheckable.
	PromptTokPerSec float64 `json:"prompt_tok_per_sec,omitempty"`

	GenTokens   int     `json:"gen_tokens_per_stream"`
	LlamaCppRef string  `json:"llama_cpp_ref"`
	TookSeconds float64 `json:"took_seconds"`
	Note        string  `json:"note,omitempty"`
}

// RunSelftest measures the engine at its configured stream count, briefly, before it serves.
//
// This runs against the model already loaded rather than starting a second copy of anything.
// The cost is a few seconds once, against not knowing whether a deployment is on a healthy
// card, on a library build whose decode cost changed, or batching at all.
func RunSelftest(ctx context.Context, eng *engine.Engine, streams int, genTokens int, llamaCppRef string) Selftest {
	st := Selftest{Streams: streams, LlamaCppRef: llamaCppRef}
	if streams < 1 {
		st.Streams, streams = 1, 1
	}
	if genTokens < 2 {
		genTokens = 32
	}
	st.GenTokens = genTokens
	start := time.Now()

	res, err := Run(ctx, eng, Config{
		Prompt:  selftestPrompt,
		Streams: streams,
		Tokens:  genTokens,
		Warmup:  4,
	})
	if err != nil {
		st.Note = fmt.Sprintf("selftest failed: %v", err)
		st.TookSeconds = time.Since(start).Seconds()
		return st
	}

	st.AggregateTokPerSec = res.DecodeTokPerSec
	st.PerStreamTokPerSec = res.PerStreamTokPerSec
	st.TokensPerPass = res.TokensPerPass
	if res.Library != nil {
		st.PromptTokPerSec = res.Library.PromptTokPerSec
	}
	st.TookSeconds = time.Since(start).Seconds()

	// The failure worth naming: the herd formed and bought nothing. Aggregate throughput can
	// look reasonable while every extra stream costs a full pass, and no serving metric shows
	// it — only tokens per pass against the stream count does.
	if streams > 1 && res.TokensPerPass > 0 && res.TokensPerPass < float64(streams)*0.75 {
		st.Note = fmt.Sprintf("only %.2f tokens per pass across %d streams — "+
			"the herd is not batching, so concurrency is costing passes rather than sharing them",
			res.TokensPerPass, streams)
	}
	return st
}

// selftestPrompt is fixed so the figure is comparable between restarts and between
// deployments. A varying prompt would make every reading its own experiment.
const selftestPrompt = "Explain in detail how ocean currents move heat around the planet, " +
	"and why that matters for regional climate."
