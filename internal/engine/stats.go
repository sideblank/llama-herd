// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package engine

import "sync/atomic"

// Stats is a snapshot of one engine's capacity and cumulative work.
//
// Capacity numbers are what an operator needs to answer "can this host take more load"
// without guessing from latency: slots in use against slots available, and how deep the
// queue is behind them.
type Stats struct {
	// StreamsMax is how many concurrent generations this model can serve.
	StreamsMax int `json:"streams_max"`
	// StreamsActive is how many are decoding right now.
	StreamsActive int `json:"streams_active"`
	// Queued is how many requests are waiting for a slot. A persistently non-zero queue
	// means the model is saturated, which latency alone reports only indirectly.
	Queued int `json:"queued"`

	// ContextTotal is the context shared across all streams; ContextPerStream is what one
	// request can occupy.
	ContextTotal     uint32 `json:"context_total"`
	ContextPerStream uint32 `json:"context_per_stream"`

	// Cumulative counters since start.
	RequestsTotal   uint64 `json:"requests_total"`
	RequestsFailed  uint64 `json:"requests_failed"`
	TokensGenerated uint64 `json:"tokens_generated"`
	PromptTokens    uint64 `json:"prompt_tokens"`
	// EvictionsTotal counts streams ended because the KV cache filled. A rising count
	// means the context budget is over-committed for the offered load.
	EvictionsTotal uint64 `json:"evictions_total"`
}

// counters holds the atomics behind Stats.
type counters struct {
	requests  atomic.Uint64
	failed    atomic.Uint64
	tokens    atomic.Uint64
	prompt    atomic.Uint64
	evictions atomic.Uint64
	active    atomic.Int64
}

// Stats returns a snapshot. Safe to call from any goroutine.
func (e *Engine) Stats() Stats {
	e.mu.Lock()
	queued := len(e.queue)
	free := len(e.free)
	e.mu.Unlock()

	max := int(e.be.NSeqMax())
	active := int(e.c.active.Load())
	// free and active are sampled separately and can disagree by one mid-tick; report the
	// derived value rather than letting a transient show as impossible capacity.
	if active > max {
		active = max
	}
	_ = free

	return Stats{
		StreamsMax:       max,
		StreamsActive:    active,
		Queued:           queued,
		ContextTotal:     e.be.NCtx(),
		ContextPerStream: e.be.NCtxSeq(),
		RequestsTotal:    e.c.requests.Load(),
		RequestsFailed:   e.c.failed.Load(),
		TokensGenerated:  e.c.tokens.Load(),
		PromptTokens:     e.c.prompt.Load(),
		EvictionsTotal:   e.c.evictions.Load(),
	}
}
