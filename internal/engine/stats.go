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
	// DecodePasses counts forward passes, prefill included.
	//
	// Do not read tokens divided by passes as an acceptance rate. A pass can carry prefill
	// for one stream and decode for another, and a prefill pass produces no tokens at all,
	// so the ratio moves with prompt length and stream mix rather than with speculation —
	// it was misleading enough to read below one on a long prompt with a short answer.
	// Use DraftsProposed and DraftsAccepted, which mean exactly one thing.
	DecodePasses uint64 `json:"decode_passes"`

	// DraftsProposed and DraftsAccepted count speculative tokens offered and kept.
	//
	// Their ratio is the acceptance rate, unaffected by prompt length, stream count, host
	// contention or batching. Zero proposed means no drafter is running; proposals with no
	// acceptances means the draft source is spending batch space for nothing.
	DraftsProposed uint64 `json:"drafts_proposed"`
	DraftsAccepted uint64 `json:"drafts_accepted"`

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
	// passes counts forward passes, counted here because the inference library's own
	// evaluation counter only increments for single-token batches.
	passes    atomic.Uint64
	proposed  atomic.Uint64
	acceptedD atomic.Uint64
	active    atomic.Int64
}

// AcceptanceRate is the fraction of proposed draft tokens the target kept, or 0 when nothing
// was proposed. This is the number that says whether speculation earns its batch space.
func (s Stats) AcceptanceRate() float64 {
	if s.DraftsProposed == 0 {
		return 0
	}
	return float64(s.DraftsAccepted) / float64(s.DraftsProposed)
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
		DecodePasses:     e.c.passes.Load(),
		DraftsProposed:   e.c.proposed.Load(),
		DraftsAccepted:   e.c.acceptedD.Load(),
	}
}
