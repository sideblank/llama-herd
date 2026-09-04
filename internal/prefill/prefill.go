// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package prefill orchestrates a wide fan-out of pre-tokenized chunks across the herd.
//
// The parallelism lives inside the batch, not across goroutines. libllama's context is not safe for
// concurrent use — `internal/llama/binding.go` states it, and `Runner` repeats it — so exactly one
// goroutine may call into the engine. Forty-eight goroutines each invoking a decode would race
// inside llama.cpp and CUDA, and would also defeat the thing that makes this engine fast: 48
// sequences ride ONE decode pass, and 48 concurrent callers would produce 48 passes of one sequence
// each.
//
// So the worker pool sits on the host side of the engine boundary — tokenizing, assembling buffers,
// parsing results — and submission is serialised through the single owner goroutine.
package prefill

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"
)

// Work is one pre-tokenized chunk.
//
// Tokens are prepared before the fan-out, deliberately. Tokenizing inside the workers puts CGO
// allocation and GC pressure in the loop precisely when the GPU is waiting to be fed.
type Work struct {
	ID     int
	Header []int32 // pre-tokenized shared symbol table, identical across chunks
	Tokens []int32
}

// Result is one chunk's outcome.
type Result struct {
	ID   int
	Data []byte
	Err  error
	Took time.Duration
}

// OK reports whether this chunk produced usable output.
func (r Result) OK() bool { return r.Err == nil }

// Engine is the single-owner boundary to the inference context.
//
// One implementation drives one llama context and is called from one goroutine only. It is an
// interface so the orchestration can be tested without a GPU, which is most of why this package can
// have tests at all.
type Engine interface {
	// Submit evaluates one sequence. Called from exactly one goroutine, never concurrently.
	Submit(ctx context.Context, seq int, tokens []int32) ([]byte, error)
}

// Policy decides what a chunk failure does to the run.
type Policy int

const (
	// Continue records the failure and keeps going. Correct for judging, where chunks are
	// independent: one bad chunk out of 48 must not discard the 47 good ones, and the gap is
	// reported as coverage rather than thrown away.
	Continue Policy = iota
	// FailFast cancels the run on the first error. Correct for generation, where a failed tier-0
	// unit poisons everything downstream and continuing spends GPU time on work that cannot be
	// right.
	FailFast
)

// Run is the outcome of a fan-out.
type Run struct {
	Results []Result
	Wall    time.Duration
	// Failed lists chunks that errored. Kept separate and never silently merged: an answer
	// assembled from the chunks that happened to succeed is an answer about a different input.
	Failed []Result
	// Cancelled is set when FailFast stopped the run early, so a short result set is not
	// mistaken for a complete one.
	Cancelled bool
}

// Complete reports whether every chunk produced output.
func (r Run) Complete() bool { return len(r.Failed) == 0 && !r.Cancelled }

// Orchestrator fans chunks out through one engine owner.
type Orchestrator struct {
	eng Engine
	// streams bounds how many sequences are in flight, which should be the deployment's stream
	// budget. Beyond it, work queues without adding parallelism.
	streams int
	// Policy selects failure behaviour; see Policy.
	Policy Policy
	// LockThread pins the engine-owner goroutine to one OS thread.
	//
	// Not for the reason usually given: a goroutine is already pinned to its thread for the
	// duration of any single cgo call, so this does nothing within one call. It matters because
	// CUDA context state is THREAD-LOCAL, so consecutive calls landing on different OS threads
	// re-enter a different CUDA context. One long-lived locked thread keeps that state stable.
	LockThread bool
}

func New(eng Engine, streams int) *Orchestrator {
	return &Orchestrator{eng: eng, streams: streams}
}

type submission struct {
	work Work
	seq  int
	out  chan Result
}

// Process runs every chunk, submitting through a single engine-owner goroutine.
func (o *Orchestrator) Process(ctx context.Context, chunks []Work) (Run, error) {
	if len(chunks) == 0 {
		return Run{}, nil
	}
	if o.streams < 1 {
		return Run{}, fmt.Errorf("prefill: streams must be at least 1, got %d", o.streams)
	}
	// Guard the index-assignment pattern. Writing results[w.ID] is genuinely lock-free — distinct
	// elements, and Wait supplies the happens-before edge for the reader — but it is only safe
	// when IDs are exactly 0..n-1. A subset of chunks carrying their original ids (12, 13, 30)
	// would write out of range or, worse, into another chunk's slot.
	seen := make([]bool, len(chunks))
	for _, w := range chunks {
		if w.ID < 0 || w.ID >= len(chunks) {
			return Run{}, fmt.Errorf("prefill: chunk id %d is outside 0..%d — index assignment "+
				"needs dense ids; renumber before dispatch and carry the original id alongside",
				w.ID, len(chunks)-1)
		}
		if seen[w.ID] {
			return Run{}, fmt.Errorf("prefill: duplicate chunk id %d — one result would silently "+
				"overwrite the other", w.ID)
		}
		seen[w.ID] = true
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	results := make([]Result, len(chunks))
	subs := make(chan submission)

	// The single engine owner. Every call into the context happens here, in submission order.
	var owner sync.WaitGroup
	owner.Add(1)
	go func() {
		defer owner.Done()
		if o.LockThread {
			runtime.LockOSThread()
			// Unlocked on return so the OS thread is reused rather than destroyed. Omitting
			// this leaks a thread per run.
			defer runtime.UnlockOSThread()
		}
		for s := range subs {
			data, err := o.eng.Submit(runCtx, s.seq, s.work.Tokens)
			s.out <- Result{ID: s.work.ID, Data: data, Err: err}
		}
	}()

	var (
		wg        sync.WaitGroup
		mu        sync.Mutex
		firstErr  error
		cancelled bool
	)
	sem := make(chan struct{}, o.streams)
	start := time.Now()

	for i, w := range chunks {
		select {
		case <-runCtx.Done():
			mu.Lock()
			cancelled = true
			mu.Unlock()
		default:
		}
		mu.Lock()
		stop := cancelled
		mu.Unlock()
		if stop {
			break
		}

		wg.Add(1)
		go func(w Work, seq int) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-runCtx.Done():
				results[w.ID] = Result{ID: w.ID, Err: runCtx.Err()}
				return
			}

			// Host-side assembly runs here, off the owner goroutine, so the engine is never
			// waiting on buffer building. The header is identical across chunks and is the
			// KV-prefix-reuse candidate; it is still concatenated per chunk because nothing
			// below this layer shares a prefix yet.
			buf := make([]int32, 0, len(w.Header)+len(w.Tokens))
			buf = append(buf, w.Header...)
			buf = append(buf, w.Tokens...)
			job := w
			job.Tokens = buf

			out := make(chan Result, 1)
			t0 := time.Now()
			select {
			case subs <- submission{work: job, seq: seq, out: out}:
			case <-runCtx.Done():
				results[w.ID] = Result{ID: w.ID, Err: runCtx.Err()}
				return
			}
			var r Result
			select {
			case r = <-out:
			case <-runCtx.Done():
				results[w.ID] = Result{ID: w.ID, Err: runCtx.Err()}
				return
			}
			r.Took = time.Since(t0)
			results[w.ID] = r

			if r.Err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("chunk %d: %w", w.ID, r.Err)
				}
				if o.Policy == FailFast {
					cancelled = true
					mu.Unlock()
					cancel()
					return
				}
				mu.Unlock()
			}
		}(w, i%o.streams)
	}

	wg.Wait()
	close(subs)
	owner.Wait()

	run := Run{Wall: time.Since(start), Cancelled: cancelled}
	for _, r := range results {
		run.Results = append(run.Results, r)
		if !r.OK() {
			run.Failed = append(run.Failed, r)
		}
	}
	sort.Slice(run.Failed, func(i, j int) bool { return run.Failed[i].ID < run.Failed[j].ID })

	// FailFast surfaces the error; Continue returns the partial run and lets the caller decide,
	// because for judging the coverage of a partial run is itself the finding.
	if o.Policy == FailFast && firstErr != nil {
		return run, firstErr
	}
	return run, nil
}

// ErrNoWork is returned when a run is asked for nothing.
var ErrNoWork = errors.New("prefill: no chunks provided")
