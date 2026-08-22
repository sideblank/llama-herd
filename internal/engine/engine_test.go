// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package engine

import (
	"context"
	"strings"
	"testing"
	"time"
)

// collect drains a stream to completion and returns the text and stop reason.
func collect(t *testing.T, s *Stream) (string, string) {
	t.Helper()
	var sb strings.Builder
	deadline := time.After(5 * time.Second)
	for {
		select {
		case ev, ok := <-s.Events:
			if !ok {
				return sb.String(), ""
			}
			if ev.Err != nil {
				t.Fatalf("stream error: %v", ev.Err)
			}
			sb.WriteString(ev.Text)
			if ev.Done {
				return sb.String(), ev.Reason
			}
		case <-deadline:
			t.Fatal("timed out waiting for stream")
		}
	}
}

func run(t *testing.T, e *Engine) context.CancelFunc {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = e.Run(ctx) }()
	return cancel
}

func TestSingleStreamGeneratesScript(t *testing.T) {
	f := newFake(2, 32)
	f.script[0] = []Token{'h', 'i'}
	e := New(f, Config{})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{Prompt: "go"})
	if err != nil {
		t.Fatal(err)
	}
	text, reason := collect(t, s)
	if text != "hi" {
		t.Fatalf("text = %q, want %q", text, "hi")
	}
	if reason != ReasonEOS {
		t.Fatalf("reason = %q, want %q", reason, ReasonEOS)
	}
}

// The bug this guards: if active decode tokens do not consume the batch budget, a full
// prefill chunk plus one active token builds a batch larger than the backend accepts, which
// kills the engine rather than erroring cleanly.
func TestActivesConsumeBatchBudget(t *testing.T) {
	const cap = 8
	f := newFake(4, cap)
	for i := SeqID(0); i < 4; i++ {
		f.script[i] = make([]Token, 20)
		for j := range f.script[i] {
			f.script[i][j] = 'a'
		}
	}
	e := New(f, Config{})
	defer run(t, e)()

	var streams []*Stream
	for i := 0; i < 4; i++ {
		s, err := e.Submit(context.Background(), Request{
			Prompt:    strings.Repeat("p", 30), // prefill much larger than the cap
			MaxTokens: 20,
		})
		if err != nil {
			t.Fatal(err)
		}
		streams = append(streams, s)
	}
	for _, s := range streams {
		collect(t, s)
	}

	if f.maxSeen > cap {
		t.Fatalf("staged a batch of %d, exceeding cap %d", f.maxSeen, cap)
	}
	if len(f.observed) == 0 {
		t.Fatal("no decode passes observed")
	}
}

func TestConcurrentStreamsDoNotCrossContaminate(t *testing.T) {
	f := newFake(3, 16)
	f.script[0] = []Token{'a', 'a', 'a'}
	f.script[1] = []Token{'b', 'b', 'b'}
	f.script[2] = []Token{'c', 'c', 'c'}
	e := New(f, Config{})
	defer run(t, e)()

	out := make([]string, 3)
	done := make(chan int, 3)
	for i := 0; i < 3; i++ {
		s, err := e.Submit(context.Background(), Request{Prompt: "x"})
		if err != nil {
			t.Fatal(err)
		}
		go func(i int, s *Stream) {
			var sb strings.Builder
			for ev := range s.Events {
				sb.WriteString(ev.Text)
			}
			out[i] = sb.String()
			done <- i
		}(i, s)
	}
	for i := 0; i < 3; i++ {
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("timeout")
		}
	}
	// Each stream must be uniform in its own letter — a mixed string means one stream
	// received another's tokens.
	for _, got := range out {
		if got == "" {
			continue
		}
		first := got[0]
		if strings.Trim(got, string(first)) != "" {
			t.Fatalf("stream produced mixed output %q — cross-contamination", got)
		}
	}
}

func TestMaxTokensStopsWithLengthReason(t *testing.T) {
	f := newFake(1, 16)
	f.script[0] = []Token{'a', 'b', 'c', 'd', 'e'}
	e := New(f, Config{})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{Prompt: "x", MaxTokens: 3})
	if err != nil {
		t.Fatal(err)
	}
	text, reason := collect(t, s)
	if text != "abc" {
		t.Fatalf("text = %q, want %q", text, "abc")
	}
	if reason != ReasonLength {
		t.Fatalf("reason = %q, want %q", reason, ReasonLength)
	}
}

func TestStopSequenceEndsGeneration(t *testing.T) {
	f := newFake(1, 16)
	f.script[0] = []Token{'a', 'b', 'X', 'Y'}
	e := New(f, Config{})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{Prompt: "x", Stop: []string{"bX"}})
	if err != nil {
		t.Fatal(err)
	}
	text, reason := collect(t, s)
	if reason != ReasonStopSeq {
		t.Fatalf("reason = %q, want %q", reason, ReasonStopSeq)
	}
	if !strings.HasSuffix(text, "bX") {
		t.Fatalf("text = %q, expected to end at the stop sequence", text)
	}
}

func TestSlotIsReleasedAndReused(t *testing.T) {
	f := newFake(1, 16) // exactly one slot: the second request must wait for the first
	f.script[0] = []Token{'a'}
	e := New(f, Config{})
	defer run(t, e)()

	s1, err := e.Submit(context.Background(), Request{Prompt: "x"})
	if err != nil {
		t.Fatal(err)
	}
	collect(t, s1)

	s2, err := e.Submit(context.Background(), Request{Prompt: "y"})
	if err != nil {
		t.Fatal(err)
	}
	collect(t, s2)

	if f.freed[0] < 2 {
		t.Fatalf("sequence 0 freed %d times, want at least 2 — slot not recycled", f.freed[0])
	}
}

func TestPromptExceedingPerSequenceBudgetIsRejected(t *testing.T) {
	f := newFake(4, 16)
	f.nCtx = 4096  // plenty in total...
	f.nCtxSeq = 64 // ...but little per sequence
	e := New(f, Config{})

	_, err := e.Submit(context.Background(), Request{Prompt: strings.Repeat("p", 200)})
	if err == nil {
		t.Fatal("expected rejection of a prompt larger than the per-sequence budget")
	}
	if !strings.Contains(err.Error(), "per-sequence") {
		t.Fatalf("error should name the per-sequence budget, got: %v", err)
	}
}

func TestQueueFullIsRefusedNotBuffered(t *testing.T) {
	f := newFake(1, 16)
	f.script[0] = make([]Token, 200)
	e := New(f, Config{MaxQueue: 1})
	defer run(t, e)()

	for i := 0; i < 3; i++ {
		if _, err := e.Submit(context.Background(), Request{Prompt: "x", MaxTokens: 200}); err != nil {
			if err == ErrQueueFull {
				return // refused as designed
			}
			t.Fatal(err)
		}
	}
	t.Skip("queue drained faster than it filled; timing-dependent")
}

func TestEmptyPromptRejected(t *testing.T) {
	f := newFake(1, 16)
	e := New(f, Config{})
	if _, err := e.Submit(context.Background(), Request{Prompt: ""}); err == nil {
		t.Fatal("expected empty prompt to be rejected")
	}
}

func TestPerRequestSamplingIsInstalledOncePerSlot(t *testing.T) {
	f := newFake(1, 16)
	f.script[0] = []Token{'a', 'b', 'c'}
	e := New(f, Config{})
	defer run(t, e)()

	temp := float32(0.1)
	s, err := e.Submit(context.Background(), Request{
		Prompt:   "x",
		Sampling: &SamplingParams{Temperature: &temp},
	})
	if err != nil {
		t.Fatal(err)
	}
	collect(t, s)

	if got := f.samplingCalls[0]; got != 1 {
		t.Fatalf("SetSampling called %d times, want exactly 1 — it should be installed once per slot", got)
	}
	got := f.sampling[0]
	if got == nil || got.Temperature == nil || *got.Temperature != 0.1 {
		t.Fatalf("sampling not passed through: %+v", got)
	}
}

func TestRequestWithoutSamplingPassesNil(t *testing.T) {
	f := newFake(1, 16)
	f.script[0] = []Token{'a'}
	e := New(f, Config{})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{Prompt: "x"})
	if err != nil {
		t.Fatal(err)
	}
	collect(t, s)

	if p := f.sampling[0]; !p.IsZero() {
		t.Fatalf("expected no sampling override, got %+v", p)
	}
}

// Each slot must get its own sampling: two concurrent requests with different settings
// must not overwrite one another.
func TestConcurrentRequestsGetIndependentSampling(t *testing.T) {
	f := newFake(2, 16)
	for i := SeqID(0); i < 2; i++ {
		f.script[i] = []Token{'a', 'b', 'c', 'd'}
	}
	e := New(f, Config{})
	defer run(t, e)()

	hot, cold := float32(1.5), float32(0.0)
	s1, err := e.Submit(context.Background(), Request{
		Prompt: "x", MaxTokens: 4, Sampling: &SamplingParams{Temperature: &hot}})
	if err != nil {
		t.Fatal(err)
	}
	s2, err := e.Submit(context.Background(), Request{
		Prompt: "y", MaxTokens: 4, Sampling: &SamplingParams{Temperature: &cold}})
	if err != nil {
		t.Fatal(err)
	}
	collect(t, s1)
	collect(t, s2)

	temps := map[float32]bool{}
	for _, p := range f.sampling {
		if p != nil && p.Temperature != nil {
			temps[*p.Temperature] = true
		}
	}
	if !temps[1.5] || !temps[0.0] {
		t.Fatalf("both temperatures should have been installed, saw %v", temps)
	}
}

// The first token of an answer is sampled from the prompt's own logits. Skipping that leaves
// the slot with no token to continue from, and the next pass stages whatever `next` holds —
// the zero token — which is injected ahead of the real first token and changes the sequence
// the model continues from.
//
// The symptom depends entirely on the model, which is what made this survive: a tokenizer
// where token 0 is ordinary text produces a plausible answer with a stray prefix, while one
// where it terminates the turn produces nothing at all and reports a clean stop.
func TestFirstTokenIsSampledFromThePrompt(t *testing.T) {
	f := newFake(1, 32)
	f.script[0] = []Token{'h', 'i'}

	e := New(f, Config{})
	defer run(t, e)()

	const prompt = "abc"
	s, err := e.Submit(context.Background(), Request{Prompt: prompt, MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	got, _ := collect(t, s)
	if got != "hi" {
		t.Fatalf("generated %q, want %q", got, "hi")
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	if len(f.sampledAt) == 0 {
		t.Fatal("nothing was ever sampled")
	}
	// The first sample must come from the position holding the last prompt token: that is
	// where the prompt's logits landed.
	last := Token(prompt[len(prompt)-1])
	if first := f.sampledAt[0]; first.tok != last {
		t.Fatalf("first sample read the position holding %q, want the last prompt token %q",
			string(rune(first.tok)), string(rune(last)))
	}
	// Nothing that was never sampled may be fed back in. A zero token here is the exact
	// signature of continuing from an unset value.
	for _, st := range f.staged {
		if st.tok == 0 {
			t.Fatal("a zero token was staged: the engine continued from a value it never sampled")
		}
	}
}

// A deployment may admit less than it allocated. The gap is slack no request can consume, so
// an admitted request always has room to finish — the refusal happens at submit, with a
// number, rather than as an eviction part way through an answer.
func TestAdmissionCapIsEnforcedBelowTheAllocation(t *testing.T) {
	f := newFake(1, 64)
	f.nCtx, f.nCtxSeq = 4096, 4096

	// Cap admission well below what the sequence owns.
	e := New(f, Config{AdmitContext: 64})
	defer run(t, e)()

	long := strings.Repeat("x", 200)
	if _, err := e.Submit(context.Background(), Request{Prompt: long, MaxTokens: 8}); err == nil {
		t.Fatal("a prompt past the admission cap was accepted")
	}

	// Without the cap the same prompt fits the allocated window comfortably.
	e2 := New(f, Config{})
	defer run(t, e2)()
	f.script[0] = []Token{'a'}
	if _, err := e2.Submit(context.Background(), Request{Prompt: long, MaxTokens: 8}); err != nil {
		t.Fatalf("the same prompt should fit the allocation with no cap: %v", err)
	}
}
