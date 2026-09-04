// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"
)

// Request is one unit of work for one stream.
type Request struct {
	// Chunk identifies which piece this is, and is carried through to the result so a response
	// can never be attributed to the wrong part of the document.
	Chunk int
	// Prompt is the full text for this stream: skeleton, content, instruction.
	Prompt string
	// Grammar constrains the response, when the caller wants a shape code can merge.
	Grammar string
	// MaxTokens bounds the response. Digests are meant to be small.
	MaxTokens int
}

// Runner executes one request. Implemented over HTTP against a llama-herd, or by a fake in tests.
//
// An interface rather than a concrete client because the layer is a client of the engine's API,
// not part of it — and because dispatch logic is worth testing without a GPU.
type Runner interface {
	Run(ctx context.Context, req Request) (string, error)
}

// Result is one stream's outcome.
type Result struct {
	Chunk int
	Text  string
	Err   error
	// Took is wall time for this request. Recorded because the barrier waits for the slowest,
	// so the spread across chunks is the cost of the fan-out and it is otherwise invisible.
	Took time.Duration
}

// Batch is the outcome of dispatching many requests.
type Batch struct {
	Results []Result
	// Failed lists chunks that did not produce a response.
	//
	// Kept separate and never merged into Results silently: an answer assembled from the chunks
	// that happened to succeed is an answer about a different document, and nothing downstream
	// can tell unless the gap is named here.
	Failed []Result
	// Wall is submit of the first to completion of the last.
	Wall time.Duration
	// Slowest and Median bound the straggler cost. The barrier waits for the slowest, so a wide
	// spread means the batch drained while one stream finished alone — which is the predicted
	// weak point of fanning out and the number that shows it.
	Slowest, Median time.Duration
}

// Complete reports whether every request produced a response.
func (b Batch) Complete() bool { return len(b.Failed) == 0 }

// StragglerRatio is slowest over median. Near 1 means the streams finished together and the batch
// stayed full; large means the last stream decoded largely alone, at single-stream speed, while the
// rest of the herd sat idle.
func (b Batch) StragglerRatio() float64 {
	if b.Median <= 0 {
		return 0
	}
	return float64(b.Slowest) / float64(b.Median)
}

// Dispatch runs requests concurrently, bounded by the stream budget.
//
// Bounded deliberately: submitting more than the deployment can run does not add parallelism, it
// adds queueing — and queued requests still hold their place in the caller's wait. The concurrency
// limit is what the engine can actually serve at once.
//
// Every request produces a Result, success or failure. A chunk that errors is recorded rather than
// dropped, because the alternative is an answer about a document with a hole in it and no sign that
// anything is missing.
func Dispatch(ctx context.Context, r Runner, reqs []Request, concurrency int) Batch {
	if concurrency < 1 {
		concurrency = 1
	}
	out := make([]Result, len(reqs))
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	start := time.Now()
	for i, req := range reqs {
		wg.Add(1)
		go func(i int, req Request) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			// Check cancellation after acquiring a slot: a caller that has gone away should not
			// have its queued work started.
			if err := ctx.Err(); err != nil {
				out[i] = Result{Chunk: req.Chunk, Err: err}
				return
			}
			t0 := time.Now()
			text, err := r.Run(ctx, req)
			out[i] = Result{Chunk: req.Chunk, Text: text, Err: err, Took: time.Since(t0)}
		}(i, req)
	}
	wg.Wait()

	b := Batch{Wall: time.Since(start)}
	var took []time.Duration
	for _, res := range out {
		if res.Err != nil {
			b.Failed = append(b.Failed, res)
			continue
		}
		b.Results = append(b.Results, res)
		took = append(took, res.Took)
	}
	// Document order, never completion order — downstream assembly depends on it.
	sort.SliceStable(b.Results, func(i, j int) bool { return b.Results[i].Chunk < b.Results[j].Chunk })
	sort.SliceStable(b.Failed, func(i, j int) bool { return b.Failed[i].Chunk < b.Failed[j].Chunk })

	if len(took) > 0 {
		sort.Slice(took, func(i, j int) bool { return took[i] < took[j] })
		b.Median = took[len(took)/2]
		b.Slowest = took[len(took)-1]
	}
	return b
}

// Digests parses every successful result, refusing the whole batch if any chunk is missing.
//
// All-or-nothing on purpose. Partial input to a merge produces an answer that reads as complete and
// is not, which is the failure this pipeline exists to prevent — so the gap is raised here, where
// it can still be retried or reported, rather than downstream where it is invisible.
func (b Batch) Digests() ([]Digest, error) {
	if !b.Complete() {
		var chunks []int
		for _, f := range b.Failed {
			chunks = append(chunks, f.Chunk)
		}
		return nil, fmt.Errorf("vcontext: %d of %d chunks failed (%v) — merging the rest would "+
			"answer about a document with holes in it",
			len(b.Failed), len(b.Failed)+len(b.Results), chunks)
	}
	out := make([]Digest, 0, len(b.Results))
	for _, res := range b.Results {
		d, err := ParseDigest(res.Chunk, res.Text)
		if err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, nil
}
