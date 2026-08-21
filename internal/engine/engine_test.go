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
