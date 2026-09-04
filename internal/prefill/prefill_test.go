// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package prefill

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type fakeEngine struct {
	mu       sync.Mutex
	inFlight int32
	maxSeen  int32
	calls    int
	seqs     []int
	lens     []int
	fail     map[int]bool // by first token, used to target a chunk
	delay    time.Duration
}

func (f *fakeEngine) Submit(ctx context.Context, seq int, tokens []int32) ([]byte, error) {
	// The contract this whole package exists to honour: exactly one goroutine may be inside the
	// engine at a time. Counting rather than asserting, so the failure is a number.
	n := atomic.AddInt32(&f.inFlight, 1)
	for {
		old := atomic.LoadInt32(&f.maxSeen)
		if n <= old || atomic.CompareAndSwapInt32(&f.maxSeen, old, n) {
			break
		}
	}
	if f.delay > 0 {
		time.Sleep(f.delay)
	}
	f.mu.Lock()
	f.calls++
	f.seqs = append(f.seqs, seq)
	f.lens = append(f.lens, len(tokens))
	shouldFail := len(tokens) > 0 && f.fail[int(tokens[len(tokens)-1])]
	f.mu.Unlock()
	atomic.AddInt32(&f.inFlight, -1)
	if shouldFail {
		return nil, fmt.Errorf("engine refused")
	}
	return []byte("ok"), nil
}

func work(n int, header []int32) []Work {
	var out []Work
	for i := 0; i < n; i++ {
		out = append(out, Work{ID: i, Header: header, Tokens: []int32{int32(i)}})
	}
	return out
}

// The load-bearing test.
func TestEngineIsNeverEnteredConcurrently(t *testing.T) {
	f := &fakeEngine{delay: 200 * time.Microsecond}
	o := New(f, 48)
	run, err := o.Process(context.Background(), work(200, nil))
	if err != nil {
		t.Fatal(err)
	}
	if !run.Complete() {
		t.Fatalf("expected a complete run, failed: %v", run.Failed)
	}
	if got := atomic.LoadInt32(&f.maxSeen); got != 1 {
		t.Fatalf("%d goroutines were inside the engine at once — libllama's context is not safe "+
			"for concurrent use, so this is a data race in llama.cpp and CUDA, not a style issue", got)
	}
	if f.calls != 200 {
		t.Fatalf("every chunk must reach the engine exactly once, got %d", f.calls)
	}
}

func TestHeaderIsPrependedToEveryChunk(t *testing.T) {
	f := &fakeEngine{}
	o := New(f, 4)
	hdr := []int32{9, 9, 9}
	if _, err := o.Process(context.Background(), work(6, hdr)); err != nil {
		t.Fatal(err)
	}
	for _, l := range f.lens {
		if l != len(hdr)+1 {
			t.Fatalf("each submission is header+body = %d tokens, got %d", len(hdr)+1, l)
		}
	}
}

func TestResultsLandAtTheirOwnIndex(t *testing.T) {
	f := &fakeEngine{}
	o := New(f, 8)
	run, err := o.Process(context.Background(), work(50, nil))
	if err != nil {
		t.Fatal(err)
	}
	for i, r := range run.Results {
		if r.ID != i {
			t.Fatalf("result %d carries id %d — a misattributed chunk is invisible downstream", i, r.ID)
		}
	}
}

func TestSparseIDsAreRejectedNotSilentlyMisplaced(t *testing.T) {
	f := &fakeEngine{}
	o := New(f, 4)
	// A subset of a larger document carrying original ids.
	chunks := []Work{{ID: 12, Tokens: []int32{1}}, {ID: 30, Tokens: []int32{2}}}
	if _, err := o.Process(context.Background(), chunks); err == nil {
		t.Fatal("index assignment needs dense ids; sparse ids would write out of range or into another chunk's slot")
	}
}

func TestDuplicateIDsAreRejected(t *testing.T) {
	f := &fakeEngine{}
	o := New(f, 4)
	chunks := []Work{{ID: 0, Tokens: []int32{1}}, {ID: 0, Tokens: []int32{2}}}
	if _, err := o.Process(context.Background(), chunks); err == nil {
		t.Fatal("one result would silently overwrite the other")
	}
}

func TestConcurrencyBoundIsRespected(t *testing.T) {
	f := &fakeEngine{delay: 500 * time.Microsecond}
	o := New(f, 4)
	var cur, peak int32
	// Wrap to observe host-side concurrency rather than engine concurrency.
	o.eng = engFunc(func(ctx context.Context, seq int, tok []int32) ([]byte, error) {
		n := atomic.AddInt32(&cur, 1)
		for {
			old := atomic.LoadInt32(&peak)
			if n <= old || atomic.CompareAndSwapInt32(&peak, old, n) {
				break
			}
		}
		time.Sleep(200 * time.Microsecond)
		atomic.AddInt32(&cur, -1)
		return []byte("ok"), nil
	})
	if _, err := o.Process(context.Background(), work(40, nil)); err != nil {
		t.Fatal(err)
	}
}

// Judging: one bad chunk must not discard 47 good ones.
func TestContinuePolicyPreservesGoodResults(t *testing.T) {
	f := &fakeEngine{fail: map[int]bool{7: true}}
	o := New(f, 8)
	o.Policy = Continue
	run, err := o.Process(context.Background(), work(48, nil))
	if err != nil {
		t.Fatalf("Continue must not surface the error as a run failure: %v", err)
	}
	if run.Complete() {
		t.Fatal("the run is not complete — one chunk failed")
	}
	if len(run.Failed) != 1 || run.Failed[0].ID != 7 {
		t.Fatalf("want chunk 7 failed, got %v", run.Failed)
	}
	good := 0
	for _, r := range run.Results {
		if r.OK() {
			good++
		}
	}
	if good != 47 {
		t.Fatalf("the other 47 chunks are still evidence and must survive; got %d", good)
	}
}

// Generation: a failed tier-0 unit poisons everything downstream.
func TestFailFastStopsAndSaysSo(t *testing.T) {
	f := &fakeEngine{fail: map[int]bool{3: true}, delay: 300 * time.Microsecond}
	o := New(f, 4)
	o.Policy = FailFast
	run, err := o.Process(context.Background(), work(200, nil))
	if err == nil {
		t.Fatal("FailFast must surface the error")
	}
	if !run.Cancelled {
		t.Fatal("a short result set must be marked cancelled, or it reads as a complete run")
	}
	if run.Complete() {
		t.Fatal("a cancelled run is never complete")
	}
}

func TestCancellationIsHonoured(t *testing.T) {
	f := &fakeEngine{delay: time.Millisecond}
	o := New(f, 8)
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(2 * time.Millisecond); cancel() }()
	run, _ := o.Process(ctx, work(500, nil))
	if run.Complete() {
		t.Fatal("a cancelled run must not report complete")
	}
}

func TestEmptyRun(t *testing.T) {
	run, err := New(&fakeEngine{}, 4).Process(context.Background(), nil)
	if err != nil || len(run.Results) != 0 {
		t.Fatalf("got %+v %v", run, err)
	}
}

func TestZeroStreamsRejected(t *testing.T) {
	if _, err := New(&fakeEngine{}, 0).Process(context.Background(), work(1, nil)); err == nil {
		t.Fatal("zero streams would deadlock rather than fail")
	}
}

func TestLockThreadDoesNotChangeResults(t *testing.T) {
	f := &fakeEngine{}
	o := New(f, 8)
	o.LockThread = true
	run, err := o.Process(context.Background(), work(30, nil))
	if err != nil || !run.Complete() {
		t.Fatalf("locking the owner thread must be transparent: %v %+v", err, run.Failed)
	}
}

type engFunc func(context.Context, int, []int32) ([]byte, error)

func (e engFunc) Submit(c context.Context, s int, t []int32) ([]byte, error) { return e(c, s, t) }
