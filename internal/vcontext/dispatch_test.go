// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeRunner answers deterministically and records how many ran at once.
type fakeRunner struct {
	delay    func(chunk int) time.Duration
	failOn   map[int]bool
	inFlight atomic.Int32
	peak     atomic.Int32
	mu       sync.Mutex
	order    []int
}

func (f *fakeRunner) Run(ctx context.Context, req Request) (string, error) {
	n := f.inFlight.Add(1)
	for {
		p := f.peak.Load()
		if n <= p || f.peak.CompareAndSwap(p, n) {
			break
		}
	}
	defer f.inFlight.Add(-1)

	d := time.Millisecond
	if f.delay != nil {
		d = f.delay(req.Chunk)
	}
	select {
	case <-time.After(d):
	case <-ctx.Done():
		return "", ctx.Err()
	}
	f.mu.Lock()
	f.order = append(f.order, req.Chunk)
	f.mu.Unlock()

	if f.failOn[req.Chunk] {
		return "", errors.New("stream failed")
	}
	return fmt.Sprintf(`{"chunk":%d}`, req.Chunk), nil
}

func reqs(n int) []Request {
	out := make([]Request, n)
	for i := range out {
		out[i] = Request{Chunk: i, Prompt: "x", MaxTokens: 30}
	}
	return out
}

// Submitting more than the deployment can serve does not add parallelism, it adds queueing — and
// queued work still holds the caller's wait.
func TestConcurrencyIsBounded(t *testing.T) {
	f := &fakeRunner{delay: func(int) time.Duration { return 5 * time.Millisecond }}
	Dispatch(context.Background(), f, reqs(40), 8)
	if peak := f.peak.Load(); peak > 8 {
		t.Errorf("ran %d concurrently against a limit of 8", peak)
	}
	if peak := f.peak.Load(); peak < 2 {
		t.Errorf("peak concurrency %d — dispatch did not actually parallelise", peak)
	}
}

// Results must be in document order however the streams finished, because assembly depends on it.
func TestResultsAreInDocumentOrderNotCompletionOrder(t *testing.T) {
	// Later chunks finish first, so completion order is the reverse of document order.
	f := &fakeRunner{delay: func(c int) time.Duration {
		return time.Duration(20-c) * time.Millisecond
	}}
	b := Dispatch(context.Background(), f, reqs(10), 10)
	for i, r := range b.Results {
		if r.Chunk != i {
			t.Fatalf("position %d holds chunk %d — results are in completion order", i, r.Chunk)
		}
	}
}

// A chunk that fails must be named. An answer assembled from whatever succeeded is an answer about
// a different document, and nothing downstream can tell.
func TestFailuresAreNamedNotDropped(t *testing.T) {
	f := &fakeRunner{failOn: map[int]bool{3: true, 7: true}}
	b := Dispatch(context.Background(), f, reqs(10), 4)

	if b.Complete() {
		t.Fatal("batch reported complete with two failed streams")
	}
	if len(b.Failed) != 2 {
		t.Fatalf("got %d failures, want 2", len(b.Failed))
	}
	if b.Failed[0].Chunk != 3 || b.Failed[1].Chunk != 7 {
		t.Errorf("failures = %v", b.Failed)
	}
	if len(b.Results) != 8 {
		t.Errorf("got %d successes, want 8", len(b.Results))
	}
}

// Digests must refuse a partial batch. Merging what survived produces an answer that reads as
// complete and is not.
func TestPartialBatchRefusesToProduceDigests(t *testing.T) {
	f := &fakeRunner{failOn: map[int]bool{2: true}}
	b := Dispatch(context.Background(), f, reqs(6), 6)

	_, err := b.Digests()
	if err == nil {
		t.Fatal("a batch with a failed chunk produced digests anyway")
	}
	if !contains(err.Error(), "holes") {
		t.Errorf("the refusal should say why: %v", err)
	}
}

func TestCompleteBatchParsesToDigests(t *testing.T) {
	f := &fakeRunner{}
	b := Dispatch(context.Background(), f, reqs(5), 5)
	ds, err := b.Digests()
	if err != nil {
		t.Fatal(err)
	}
	if len(ds) != 5 {
		t.Fatalf("got %d digests, want 5", len(ds))
	}
	for i, d := range ds {
		if d.Chunk != i {
			t.Errorf("digest %d carries chunk %d — attribution is wrong", i, d.Chunk)
		}
	}
}

// The barrier waits for the slowest, so a wide spread means the batch drained while one stream
// finished alone. That cost is otherwise invisible, which is why it is measured.
func TestStragglerIsVisible(t *testing.T) {
	f := &fakeRunner{delay: func(c int) time.Duration {
		if c == 9 {
			return 60 * time.Millisecond // one straggler
		}
		return 5 * time.Millisecond
	}}
	b := Dispatch(context.Background(), f, reqs(10), 10)
	if b.StragglerRatio() < 3 {
		t.Errorf("straggler ratio %.1f did not reflect a 12x outlier", b.StragglerRatio())
	}

	even := &fakeRunner{delay: func(int) time.Duration { return 5 * time.Millisecond }}
	eb := Dispatch(context.Background(), even, reqs(10), 10)
	if r := eb.StragglerRatio(); r > 2.5 {
		t.Errorf("evenly-sized work reported a straggler ratio of %.1f", r)
	}
}

// A caller that goes away must not have its queued work started.
func TestCancellationStopsQueuedWork(t *testing.T) {
	f := &fakeRunner{delay: func(int) time.Duration { return 30 * time.Millisecond }}
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(10 * time.Millisecond); cancel() }()

	b := Dispatch(ctx, f, reqs(40), 2)
	if b.Complete() {
		t.Error("a cancelled dispatch reported complete")
	}
	if len(b.Failed) == 0 {
		t.Error("cancellation produced no recorded failures")
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (func() bool {
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				return true
			}
		}
		return false
	})()
}

// The HTTP runner must surface an engine error as an error, not as empty text that a merge would
// try to parse.
func TestHTTPRunnerSurfacesEngineErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(500)
		fmt.Fprint(w, `{"error":{"message":"grammar failed to parse"}}`)
	}))
	defer srv.Close()

	r := NewHTTPRunner(srv.URL, "chat")
	_, err := r.Run(context.Background(), Request{Chunk: 4, Prompt: "x"})
	if err == nil {
		t.Fatal("an engine error was not surfaced")
	}
	if !contains(err.Error(), "grammar failed to parse") {
		t.Errorf("the engine's reason was lost: %v", err)
	}
	if !contains(err.Error(), "chunk 4") {
		t.Errorf("the error does not say which chunk: %v", err)
	}
}

// An empty choices array is what a truncated or filtered response looks like. Indexing it would
// panic where a clear error belongs.
func TestHTTPRunnerHandlesEmptyChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[]}`)
	}))
	defer srv.Close()

	r := NewHTTPRunner(srv.URL, "chat")
	if _, err := r.Run(context.Background(), Request{Chunk: 1}); err == nil {
		t.Error("empty choices must be an error, not empty text")
	}
}

// The happy path, including that the grammar and zero temperature actually reach the engine —
// chunk work must be reproducible or a merge conflict cannot be trusted.
func TestHTTPRunnerSendsGrammarAndGreedy(t *testing.T) {
	var got chatReq
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&got)
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"{\"a\":1}"}}]}`)
	}))
	defer srv.Close()

	r := NewHTTPRunner(srv.URL, "chat")
	text, err := r.Run(context.Background(), Request{
		Chunk: 0, Prompt: "extract", Grammar: `root ::= "{}"`, MaxTokens: 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	if text != `{"a":1}` {
		t.Errorf("content = %q", text)
	}
	if got.Grammar == "" {
		t.Error("the grammar did not reach the engine")
	}
	if got.Temperature != 0 {
		t.Errorf("temperature = %v, want 0 — chunk extraction must be reproducible", got.Temperature)
	}
}
