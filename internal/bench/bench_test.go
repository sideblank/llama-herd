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

func TestBusyMachineIsFlagged(t *testing.T) {
	// A benchmark on a loaded machine looks fine and is wrong, so the report must say so
	// rather than leaving the reader to guess.
	busy := Environment{LoadAvg1: 42.0, CPUs: 16}
	if !busy.Busy() {
		t.Fatal("load of 42 across 16 cores should be flagged")
	}
	idle := Environment{LoadAvg1: 0.4, CPUs: 16}
	if idle.Busy() {
		t.Fatal("an idle machine should not be flagged")
	}

	rep := &Report{Environment: busy, Results: []*Result{{Streams: 1, DecodeTokPerSec: 0.4}}}
	var sb strings.Builder
	if err := rep.WriteMarkdown(&sb); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sb.String(), "not trustworthy") {
		t.Fatal("a report from a busy machine must warn in the output itself")
	}
}

func TestLoadAverageIsReadable(t *testing.T) {
	load, cpus := LoadAverage()
	if cpus < 1 {
		t.Fatal("core count should be at least 1")
	}
	if load < 0 {
		t.Fatalf("load = %v", load)
	}
}

// Tokens per pass is how speculation becomes visible, and it is easy to misread. One pass
// serves every active stream, so the baseline is the stream count — not 1.
func TestSpeculationVerdictUsesStreamCountAsBaseline(t *testing.T) {
	cases := []struct {
		name string
		r    Result
		want string
	}{
		{"single stream, no speculation",
			Result{Streams: 1, DecodePasses: 100, TokensPerPass: 1.0}, "not active"},
		{"four streams batching normally is NOT speculation",
			Result{Streams: 4, DecodePasses: 100, TokensPerPass: 3.9}, "not active"},
		{"single stream with drafts landing",
			Result{Streams: 1, DecodePasses: 100, TokensPerPass: 1.8}, "**active**"},
		{"four streams with drafts landing",
			Result{Streams: 4, DecodePasses: 100, TokensPerPass: 6.2}, "**active**"},
		{"no passes recorded",
			Result{Streams: 2, DecodePasses: 0}, "not measured"},
	}
	for _, c := range cases {
		if got := speculationVerdict(&c.r); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}

// A deployment must not be held out of service by its own measurement. The machines where
// this runs long are exactly the degraded ones, so the budget has to cap it — and the result
// has to say that it did, rather than reporting a zero that reads as "fast" or "broken".
func TestSelftestRespectsItsBudget(t *testing.T) {
	be := enginetest.New(1, 256, "abcdefghijklmnopqrstuvwxyz")
	e := engine.New(be, engine.Config{})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _ = e.Run(ctx) }()

	// Far more tokens than the budget can cover, so the cap is what ends it rather than the
	// work finishing.
	st := RunSelftest(ctx, e, 1, 200000, "test-ref", 20*time.Millisecond)

	if st.Note == "" {
		t.Fatal("a selftest that could not complete must say so, not report zeros")
	}
	if st.TookSeconds <= 0 {
		t.Fatal("elapsed time should be reported even when the measurement did not finish")
	}
	// The reference must still identify what it was measuring, or a truncated result cannot
	// be told apart from a missing one.
	if st.LlamaCppRef != "test-ref" || st.Streams != 1 {
		t.Fatalf("truncated selftest lost its provenance: %+v", st)
	}
}

// amortFake serves N streams at a fixed cost per PASS, so concurrency genuinely pays — the
// arrangement the herd is supposed to produce.
type amortFake struct {
	*enginetest.Scripted
	perPass time.Duration
}

func (a *amortFake) Decode() error {
	time.Sleep(a.perPass)
	return a.Scripted.Decode()
}

// splitFake charges per TOKEN in the batch instead of per pass — what a library does when it
// runs one forward pass per sequence. Tokens-per-pass still reads near the stream count,
// because the batch handed over is unchanged; only the cost differs.
type splitFake struct {
	*enginetest.Scripted
	perToken time.Duration
	staged   int
}

func (s *splitFake) BatchClear() { s.staged = 0; s.Scripted.BatchClear() }

func (s *splitFake) BatchAdd(tok engine.Token, pos engine.Pos, seq engine.SeqID, logits bool) error {
	if err := s.Scripted.BatchAdd(tok, pos, seq, logits); err != nil {
		return err
	}
	s.staged++
	return nil
}

func (s *splitFake) Decode() error {
	time.Sleep(time.Duration(s.staged) * s.perToken)
	return s.Scripted.Decode()
}

// The selftest has to catch a herd that forms and amortises nothing. Tokens-per-pass cannot
// see it — it counts the batch submitted, not what the library did with it — so the check is
// aggregate throughput against the same engine's single-stream rate.
func TestSelftestDetectsAHerdThatDoesNotAmortise(t *testing.T) {
	const streams = 4

	good := &amortFake{Scripted: enginetest.New(streams, 256, "abcdefghijklmnopqrstuvwxyz"),
		perPass: 4 * time.Millisecond}
	ge := engine.New(good, engine.Config{})
	gctx, gcancel := context.WithCancel(context.Background())
	defer gcancel()
	go func() { _ = ge.Run(gctx) }()
	gs := RunSelftest(gctx, ge, streams, 16, "ref", 30*time.Second)

	bad := &splitFake{Scripted: enginetest.New(streams, 256, "abcdefghijklmnopqrstuvwxyz"),
		perToken: 4 * time.Millisecond}
	be := engine.New(bad, engine.Config{})
	bctx, bcancel := context.WithCancel(context.Background())
	defer bcancel()
	go func() { _ = be.Run(bctx) }()
	bs := RunSelftest(bctx, be, streams, 16, "ref", 30*time.Second)

	t.Logf("amortising : aggregate %.1f single %.1f ratio %.2f note=%q",
		gs.AggregateTokPerSec, gs.SingleStreamTokPerSec, gs.Amortisation, gs.Note)
	t.Logf("per-sequence: aggregate %.1f single %.1f ratio %.2f note=%q",
		bs.AggregateTokPerSec, bs.SingleStreamTokPerSec, bs.Amortisation, bs.Note)

	if gs.Amortisation <= 1.0 {
		t.Errorf("a backend charging per pass should amortise, got ratio %.2f", gs.Amortisation)
	}
	if gs.Note != "" {
		t.Errorf("a healthy herd should not be flagged: %q", gs.Note)
	}
	if bs.Amortisation > 1.0 {
		t.Errorf("a backend charging per token cannot amortise, got ratio %.2f", bs.Amortisation)
	}
	if bs.Note == "" {
		t.Error("a herd that does not amortise must be reported, and tokens-per-pass will not show it")
	}
	// The point of the whole test: the metric that looks healthy in both cases.
	if bs.TokensPerPass < float64(streams)*0.75 {
		t.Fatalf("this fixture is meant to keep tokens-per-pass healthy (%.2f) while failing "+
			"to amortise — otherwise it does not prove the old check was insufficient",
			bs.TokensPerPass)
	}
}
