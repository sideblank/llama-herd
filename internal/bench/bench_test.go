// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package bench

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/sideblank/llama-herd/internal/engine"
	"github.com/sideblank/llama-herd/internal/enginetest"
)

func newEngine(t *testing.T, streams uint32, script string) (*engine.Engine, context.CancelFunc) {
	t.Helper()
	be := enginetest.New(streams, 256, script)
	e := engine.New(be, engine.Config{})
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = e.Run(ctx) }()
	return e, cancel
}

func TestRunProducesBothThroughputFigures(t *testing.T) {
	e, cancel := newEngine(t, 4, strings.Repeat("a", 64))
	defer cancel()

	res, err := Run(context.Background(), e, Config{
		Model: "m", Prompt: "hello", Streams: 4, Tokens: 16,
	})
	if err != nil {
		t.Fatal(err)
	}

	if res.TotalTokens != 64 {
		t.Errorf("total tokens = %d, want 64", res.TotalTokens)
	}
	if res.Failures != 0 {
		t.Fatalf("failures = %d: %+v", res.Failures, res.Streams_)
	}
	if res.DecodeTokPerSec <= 0 {
		t.Error("decode rate should be positive")
	}
	if res.EndToEndTokPerSec <= 0 {
		t.Error("end-to-end rate should be positive")
	}
	// Decode excludes prefill, so it must be at least the end-to-end figure. Reporting
	// them the other way round would mean the window definitions are wrong.
	if res.DecodeTokPerSec < res.EndToEndTokPerSec {
		t.Errorf("decode (%.1f) should not be below end-to-end (%.1f)",
			res.DecodeTokPerSec, res.EndToEndTokPerSec)
	}
	if res.TTFTp50 <= 0 {
		t.Error("TTFT p50 should be positive")
	}
	if res.TTFTp95 < res.TTFTp50 {
		t.Error("p95 should not be below p50")
	}
}

func TestSingleTokenRunIsRejected(t *testing.T) {
	e, cancel := newEngine(t, 1, "abc")
	defer cancel()

	if _, err := Run(context.Background(), e, Config{Model: "m", Prompt: "x", Streams: 1, Tokens: 1}); err == nil {
		t.Fatal("one token gives no decode span; the run should be rejected")
	}
}

func TestZeroStreamsRejected(t *testing.T) {
	e, cancel := newEngine(t, 1, "abc")
	defer cancel()

	if _, err := Run(context.Background(), e, Config{Model: "m", Prompt: "x", Streams: 0, Tokens: 8}); err == nil {
		t.Fatal("zero streams should be rejected")
	}
}

func TestAggregateRisesWithStreams(t *testing.T) {
	e, cancel := newEngine(t, 8, strings.Repeat("a", 128))
	defer cancel()

	one, err := Run(context.Background(), e, Config{Model: "m", Prompt: "x", Streams: 1, Tokens: 32})
	if err != nil {
		t.Fatal(err)
	}
	many, err := Run(context.Background(), e, Config{Model: "m", Prompt: "x", Streams: 8, Tokens: 32})
	if err != nil {
		t.Fatal(err)
	}

	if many.TotalTokens <= one.TotalTokens {
		t.Fatalf("8 streams produced %d tokens, 1 stream produced %d",
			many.TotalTokens, one.TotalTokens)
	}
	if many.Failures != 0 || one.Failures != 0 {
		t.Fatalf("failures: %d and %d", one.Failures, many.Failures)
	}
}

func TestWarmupIsExcludedFromCounts(t *testing.T) {
	e, cancel := newEngine(t, 2, strings.Repeat("a", 128))
	defer cancel()

	res, err := Run(context.Background(), e, Config{
		Model: "m", Prompt: "x", Streams: 2, Tokens: 10, Warmup: 5,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.TotalTokens != 20 {
		t.Fatalf("total tokens = %d, want 20 — warmup must not be counted", res.TotalTokens)
	}
}

func TestMarkdownReportStatesDefinitions(t *testing.T) {
	rep := &Report{
		Environment: Environment{
			Timestamp: "2026-08-21T00:00:00Z", Version: "v0.1.0", Commit: "abcdef1234",
			LlamaCppRef: "b10545", GoVersion: "go1.24", OS: "linux", Arch: "amd64",
			ModelName: "m", ModelPath: "/m.gguf", Context: 4096, ContextPer: 1024,
			Batch: 2048, GPULayers: -1, LoadMTP: true, PromptTokens: 12,
			Devices: []Device{{Name: "Test GPU", Type: "gpu", TotalBytes: 24 << 30}},
		},
		Results: []*Result{{
			Streams: 4, DecodeTokPerSec: 400, EndToEndTokPerSec: 350,
			PerStreamTokPerSec: 100, TTFTp50: 120 * time.Millisecond,
			TTFTp95: 200 * time.Millisecond,
		}},
	}

	var sb strings.Builder
	if err := rep.WriteMarkdown(&sb); err != nil {
		t.Fatal(err)
	}
	out := sb.String()

	// The report must be self-describing: someone reading only the output should know
	// what was measured and how to reproduce it.
	for _, want := range []string{
		"b10545", "Test GPU", "/m.gguf", "Prefill is excluded",
		"prefill included", "MTP layers loaded", "Reproduce with",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report should contain %q", want)
		}
	}
}

func TestJSONReportRoundTrips(t *testing.T) {
	rep := &Report{
		Environment: Environment{ModelName: "m", LlamaCppRef: "b10545"},
		Results:     []*Result{{Streams: 2, DecodeTokPerSec: 123.4}},
	}
	var sb strings.Builder
	if err := rep.WriteJSON(&sb); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sb.String(), "b10545") || !strings.Contains(sb.String(), "123.4") {
		t.Fatalf("json missing fields:\n%s", sb.String())
	}
}

// Per-stream must be consistent with the aggregate. Averaging each stream's own rate can
// exceed aggregate/streams when streams are staggered, which would publish more throughput
// than actually occurred.
func TestPerStreamIsConsistentWithAggregate(t *testing.T) {
	e, cancel := newEngine(t, 4, strings.Repeat("a", 128))
	defer cancel()

	res, err := Run(context.Background(), e, Config{
		Model: "m", Prompt: "x", Streams: 4, Tokens: 32,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := res.DecodeTokPerSec / 4
	if diff := res.PerStreamTokPerSec - want; diff > 0.001 || diff < -0.001 {
		t.Fatalf("per-stream %.3f is not aggregate/streams %.3f", res.PerStreamTokPerSec, want)
	}
	if res.PerStreamTokPerSec > res.DecodeTokPerSec {
		t.Fatal("per-stream must never exceed the aggregate")
	}
}
