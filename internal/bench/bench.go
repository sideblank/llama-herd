// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package bench measures throughput in a way that can be published and reproduced.
//
// Two rules shape it. Every reported number states what it measured, because "500 tok/s"
// means different things depending on whether prefill is counted. And every run records the
// configuration it ran under, because a throughput figure without the model, quantization,
// context and stream count is not a result, it is a claim.
package bench

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sideblank/llama-herd/internal/engine"
)

// LoadAverage returns the one-minute load average and the core count.
//
// A benchmark on a loaded machine is worthless and looks fine, so this is recorded with every
// run rather than left to the operator to remember.
func LoadAverage() (float64, int) {
	cpus := runtime.NumCPU()
	b, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return 0, cpus
	}
	fields := strings.Fields(string(b))
	if len(fields) == 0 {
		return 0, cpus
	}
	v, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return 0, cpus
	}
	return v, cpus
}

// LibraryPerf is the inference library's own accounting, reported alongside wall-clock
// measurement.
//
// Only the prefill figures are reported. The library's decode counter increments solely for
// single-token batches, and a batching engine submits one token per active stream, so decode
// work is attributed to prefill and the decode counter reads near zero. Upstream carries a
// FIXME acknowledging this. Decode throughput therefore comes from this package's own
// measurement, and speculation is measured from the engine's own pass counter.
type LibraryPerf struct {
	PromptTokens    int32   `json:"prompt_tokens"`
	PromptTokPerSec float64 `json:"prompt_tok_per_sec"`
	EvalCount       int32   `json:"eval_count"`
	EvalTokPerSec   float64 `json:"eval_tok_per_sec"`
	GraphReuse      int32   `json:"graph_reuse"`
	// TokensPerEval above 1.0 means a speculative head is landing drafts. Exactly 1.0 on
	// a model carrying MTP layers means the head is loaded and contributing nothing.
	TokensPerEval     float64 `json:"tokens_per_eval"`
	SpeculationActive bool    `json:"speculation_active"`
}

// PerfSource is implemented by backends that can report the library's accounting.
type PerfSource interface {
	LibraryPerf(produced uint64) LibraryPerf
	ResetLibraryPerf()
}

// Config describes one measurement.
type Config struct {
	// Model is the registered name to exercise.
	Model string
	// Prompt is sent to every stream. A shared prompt is deliberate: it keeps prefill
	// cost identical across streams so the decode comparison is clean.
	Prompt string
	// Streams is how many generations run concurrently.
	Streams int
	// Tokens is how many tokens each stream generates.
	Tokens int
	// Warmup runs first and is discarded, so one-off costs — allocation, cache
	// population, clocks ramping — do not land in the reported figure.
	Warmup int
	// Perf optionally supplies the inference library's own accounting.
	Perf any
}

// StreamResult is one generation's timings.
type StreamResult struct {
	Index int `json:"index"`
	// TTFT is time to first token: prefill plus scheduling latency.
	TTFT time.Duration `json:"ttft_ms"`
	// Decode is the span from the first token to the last, which excludes prefill.
	Decode time.Duration `json:"decode_ms"`
	// Total is submit to completion.
	Total  time.Duration `json:"total_ms"`
	Tokens int           `json:"tokens"`
	Err    string        `json:"error,omitempty"`
}

// Result is one measured configuration.
type Result struct {
	Model   string `json:"model"`
	Streams int    `json:"streams"`
	Tokens  int    `json:"tokens_per_stream"`

	// TotalTokens is every token generated across all streams.
	TotalTokens int `json:"total_tokens"`
	// Wall is submit of the first stream to completion of the last.
	Wall time.Duration `json:"wall_ms"`

	// EndToEndTokPerSec counts prefill in the elapsed time. This is what a user
	// experiences, and it is the lower of the two numbers.
	EndToEndTokPerSec float64 `json:"end_to_end_tok_per_sec"`
	// DecodeTokPerSec measures only the decode phase, from the first token emitted by
	// any stream to the last. This is the figure that reflects the decode loop itself,
	// and it is the higher of the two. Quoting it without saying so overstates.
	DecodeTokPerSec float64 `json:"decode_tok_per_sec"`

	// PerStreamTokPerSec is the aggregate divided by the stream count: the share each
	// stream effectively received.
	//
	// It is derived rather than averaged from each stream's own window on purpose.
	// Averaging individual rates measures each stream over its own span, and when
	// streams are staggered those spans are shorter than the aggregate window, so the
	// average can exceed aggregate/streams — a figure that looks like more throughput
	// than actually happened.
	PerStreamTokPerSec float64 `json:"per_stream_tok_per_sec"`
	// PerStreamMeasured is the average of each stream's own rate, kept for variance
	// analysis. See the note above before quoting it.
	PerStreamMeasured float64 `json:"per_stream_measured_tok_per_sec"`

	TTFTp50 time.Duration `json:"ttft_p50_ms"`
	TTFTp95 time.Duration `json:"ttft_p95_ms"`
	TTFTMax time.Duration `json:"ttft_max_ms"`

	Streams_ []StreamResult `json:"streams_detail,omitempty"`
	Failures int            `json:"failures"`

	// Library is the inference library's own accounting, when the backend can report it.
	Library *LibraryPerf `json:"library,omitempty"`

	// DecodePasses is forward passes taken during the measurement, counted by the engine.
	DecodePasses uint64 `json:"decode_passes"`
	// ServerTokens is what the server counted, which can differ from what the client
	// received if a connection dropped.
	ServerTokens uint64 `json:"server_tokens"`
	// Evictions during the run. Non-zero means the context budget was over-committed.
	Evictions uint64 `json:"evictions"`
	// MTPLoaded and OnGPU are reported by the server, not inferred.
	MTPLoaded bool `json:"mtp_loaded"`
	OnGPU     bool `json:"on_gpu"`

	// TokensPerPass is tokens produced per forward pass. One pass serves every active
	// stream, so this is normally about the stream count; above it means a speculative
	// head is landing drafts.
	TokensPerPass float64 `json:"tokens_per_pass"`
}

// Run measures one configuration against a live engine.
func Run(ctx context.Context, eng *engine.Engine, cfg Config) (*Result, error) {
	if cfg.Streams < 1 {
		return nil, fmt.Errorf("bench: streams must be at least 1")
	}
	if cfg.Tokens < 2 {
		// One token gives no decode span to measure: TTFT would be the whole story.
		return nil, fmt.Errorf("bench: need at least 2 tokens per stream to measure decode")
	}

	if cfg.Warmup > 0 {
		if err := warm(ctx, eng, cfg); err != nil {
			return nil, fmt.Errorf("bench: warmup: %w", err)
		}
	}
	// Clear the library's counters after warmup so its figures cover the same window as
	// the wall-clock measurement.
	if ps, ok := cfg.Perf.(PerfSource); ok && ps != nil {
		ps.ResetLibraryPerf()
	}

	// Sample the pass counter before measuring so warmup passes are excluded.
	passesBefore := eng.Stats().DecodePasses

	var (
		mu        sync.Mutex
		results   = make([]StreamResult, cfg.Streams)
		firstTok  time.Time // earliest first-token across all streams
		lastTok   time.Time // latest final-token across all streams
		totalToks int
	)

	start := time.Now()
	var wg sync.WaitGroup
	for i := 0; i < cfg.Streams; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			r := runOne(ctx, eng, cfg, i, &mu, &firstTok, &lastTok)
			mu.Lock()
			results[i] = r
			totalToks += r.Tokens
			mu.Unlock()
		}(i)
	}
	wg.Wait()
	wall := time.Since(start)

	res := &Result{
		Model:       cfg.Model,
		Streams:     cfg.Streams,
		Tokens:      cfg.Tokens,
		TotalTokens: totalToks,
		Wall:        wall,
		Streams_:    results,
	}

	var ttfts []time.Duration
	var perStream float64
	for _, r := range results {
		if r.Err != "" {
			res.Failures++
			continue
		}
		ttfts = append(ttfts, r.TTFT)
		if r.Decode > 0 && r.Tokens > 1 {
			perStream += float64(r.Tokens-1) / r.Decode.Seconds()
		}
	}

	if wall > 0 {
		res.EndToEndTokPerSec = float64(totalToks) / wall.Seconds()
	}
	// The decode window spans the first token any stream produced to the last token any
	// stream produced. Tokens before that window are prefill, and counting them here is
	// exactly the conflation this package exists to avoid.
	if !firstTok.IsZero() && lastTok.After(firstTok) {
		decoded := totalToks - (cfg.Streams - res.Failures)
		if decoded > 0 {
			res.DecodeTokPerSec = float64(decoded) / lastTok.Sub(firstTok).Seconds()
		}
	}
	if n := cfg.Streams - res.Failures; n > 0 {
		res.PerStreamMeasured = perStream / float64(n)
		res.PerStreamTokPerSec = res.DecodeTokPerSec / float64(n)
	}

	if ps, ok := cfg.Perf.(PerfSource); ok && ps != nil {
		lp := ps.LibraryPerf(uint64(totalToks))
		res.Library = &lp
	}
	if st := eng.Stats(); st.DecodePasses > passesBefore {
		res.DecodePasses = st.DecodePasses - passesBefore
		if res.DecodePasses > 0 {
			res.TokensPerPass = float64(totalToks) / float64(res.DecodePasses)
		}
	}

	res.TTFTp50 = percentile(ttfts, 0.50)
	res.TTFTp95 = percentile(ttfts, 0.95)
	res.TTFTMax = percentile(ttfts, 1.0)

	return res, nil
}

func runOne(ctx context.Context, eng *engine.Engine, cfg Config, i int,
	mu *sync.Mutex, firstTok, lastTok *time.Time) StreamResult {

	out := StreamResult{Index: i}
	submitted := time.Now()

	st, err := eng.Submit(ctx, engine.Request{
		Prompt:    cfg.Prompt,
		MaxTokens: cfg.Tokens,
	})
	if err != nil {
		out.Err = err.Error()
		return out
	}
	defer st.Close()

	var first, last time.Time
	for ev := range st.Events {
		if ev.Err != nil {
			out.Err = ev.Err.Error()
			continue
		}
		if ev.Text != "" {
			now := time.Now()
			if first.IsZero() {
				first = now
				out.TTFT = now.Sub(submitted)
			}
			last = now
			out.Tokens++
		}
	}

	out.Total = time.Since(submitted)
	if !first.IsZero() && last.After(first) {
		out.Decode = last.Sub(first)
	}

	mu.Lock()
	if !first.IsZero() && (firstTok.IsZero() || first.Before(*firstTok)) {
		*firstTok = first
	}
	if last.After(*lastTok) {
		*lastTok = last
	}
	mu.Unlock()

	return out
}

func warm(ctx context.Context, eng *engine.Engine, cfg Config) error {
	st, err := eng.Submit(ctx, engine.Request{Prompt: cfg.Prompt, MaxTokens: cfg.Warmup})
	if err != nil {
		return err
	}
	defer st.Close()
	for ev := range st.Events {
		if ev.Err != nil {
			return ev.Err
		}
	}
	return nil
}

func percentile(ds []time.Duration, p float64) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	sorted := append([]time.Duration(nil), ds...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	if p >= 1 {
		return sorted[len(sorted)-1]
	}
	idx := int(float64(len(sorted)-1) * p)
	return sorted[idx]
}
