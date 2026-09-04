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

	// PrefixSharedTotal counts prompts that adopted a resident prefix instead of computing it,
	// PrefixTokensSaved the prefill this avoided, and PrefixShareFailed the attempts that were
	// refused or did not take effect.
	//
	// The failure count is not optional bookkeeping. A failed share degrades to ordinary
	// prefill, which is correct behaviour and indistinguishable from never having tried — so
	// without this number a sharing path that has silently stopped working looks exactly like
	// one that is working on prompts with nothing in common.
	PrefixSharedTotal uint64 `json:"prefix_shared_total"`
	PrefixTokensSaved uint64 `json:"prefix_tokens_saved"`
	PrefixShareFailed uint64 `json:"prefix_share_failed"`

	// EvictionsTotal counts streams ended because the KV cache filled. A rising count
	// means the context budget is over-committed for the offered load.
	EvictionsTotal uint64 `json:"evictions_total"`

	// Where the wall clock went, cumulatively, across every pass.
	//
	// A herd short of the library's own throughput is short in exactly one of three places,
	// and the fix differs completely by which: StageNanos is this engine's Go code choosing
	// and staging a batch, DecodeNanos is the library's forward pass, HarvestNanos is
	// sampling, detokenization and delivery. Only DecodeNanos is work the library would also
	// have done, so the other two are the overhead budget, and their sum against the total is
	// the ceiling on what tuning this engine can recover.
	//
	// These are sums over concurrent streams within a pass, not per-stream times, so read
	// them as proportions of each other rather than as latencies.
	StageNanos   uint64 `json:"stage_nanos"`
	DecodeNanos  uint64 `json:"decode_nanos"`
	HarvestNanos uint64 `json:"harvest_nanos"`

	// Harvest broken down, because "harvest is expensive" does not say what to fix.
	//
	// EmitNanos in particular is not this engine's cost at all when it is large: it is the
	// caller failing to read, and optimising the sampler in response would achieve nothing.
	SampleNanos uint64 `json:"sample_nanos"`
	PieceNanos  uint64 `json:"piece_nanos"`
	EmitNanos   uint64 `json:"emit_nanos"`
}

// OverheadFraction is the share of engine time spent outside the library's forward pass —
// staging and harvesting against everything measured. It is the headroom available to this
// engine's own optimisation, and 0 when nothing has been measured yet.
func (s Stats) OverheadFraction() float64 {
	total := s.StageNanos + s.DecodeNanos + s.HarvestNanos
	if total == 0 {
		return 0
	}
	return float64(s.StageNanos+s.HarvestNanos) / float64(total)
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
	// Prefix sharing. tokensSaved is the prefill avoided by adopting a resident prefix, and
	// failed counts shares that were refused or did not take effect — kept separate because a
	// failed share is invisible otherwise: it degrades to ordinary prefill and looks identical
	// to never having tried.
	prefixShared      atomic.Uint64
	prefixTokensSaved atomic.Uint64
	prefixShareFailed atomic.Uint64
	active            atomic.Int64
	// Phase clocks, in nanoseconds. See Stats for why the split is kept.
	stageNanos   atomic.Int64
	decodeNanos  atomic.Int64
	harvestNanos atomic.Int64
	sampleNanos  atomic.Int64
	pieceNanos   atomic.Int64
	emitNanos    atomic.Int64
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
		StreamsMax:        max,
		StreamsActive:     active,
		Queued:            queued,
		ContextTotal:      e.be.NCtx(),
		ContextPerStream:  e.be.NCtxSeq(),
		RequestsTotal:     e.c.requests.Load(),
		RequestsFailed:    e.c.failed.Load(),
		TokensGenerated:   e.c.tokens.Load(),
		PromptTokens:      e.c.prompt.Load(),
		EvictionsTotal:    e.c.evictions.Load(),
		PrefixSharedTotal: e.c.prefixShared.Load(),
		PrefixTokensSaved: e.c.prefixTokensSaved.Load(),
		PrefixShareFailed: e.c.prefixShareFailed.Load(),
		DecodePasses:      e.c.passes.Load(),
		DraftsProposed:    e.c.proposed.Load(),
		DraftsAccepted:    e.c.acceptedD.Load(),
		StageNanos:        uint64(e.c.stageNanos.Load()),
		DecodeNanos:       uint64(e.c.decodeNanos.Load()),
		HarvestNanos:      uint64(e.c.harvestNanos.Load()),
		SampleNanos:       uint64(e.c.sampleNanos.Load()),
		PieceNanos:        uint64(e.c.pieceNanos.Load()),
		EmitNanos:         uint64(e.c.emitNanos.Load()),
	}
}
