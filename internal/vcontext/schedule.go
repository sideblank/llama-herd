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

// Executor runs one task. done carries the outputs of every task this one declared a dependency
// on, keyed by task id, so a dependent step receives its prerequisites' results as text.
//
// Text, not latent state: forwarding h_last between tiers is the faster path in principle, but the
// projection it needs is unresolved (#21) and the gating measurement (#19) showed injection losing
// detail the state provably holds. Building the scheduler on that would make every dependent task
// quietly lossy. Text is the honest default until the projection is measured working.
type Executor interface {
	Exec(ctx context.Context, t Task, done map[string]string) (string, error)
}

// TaskResult is one task's outcome.
type TaskResult struct {
	Task Task
	Text string
	Err  error
	Took time.Duration
	Tier int
	// Blocked is set when the task never ran because a prerequisite failed. Distinguished from
	// Err because they need opposite responses: a failed task may be worth retrying, a blocked
	// one cannot be retried until its dependency is fixed, and reporting both as "failed" hides
	// which is which.
	Blocked bool
	// BlockedBy names the failed prerequisite, transitively the nearest one.
	BlockedBy string
}

// Run is the outcome of scheduling a graph.
type Run struct {
	Results []TaskResult
	Wall    time.Duration
	// Tiers is the topological structure that was executed, kept for reporting. Execution did
	// not wait on tier boundaries — see Schedule.
	Tiers []Tier
	// CriticalPath bounds what any amount of parallelism could have achieved.
	CriticalPath []string

	// peakConcurrency is the widest overlap actually reached, recorded during the run because
	// it cannot be reconstructed from durations afterwards.
	peakConcurrency int
}

// Failed returns tasks that ran and errored.
func (r Run) Failed() []TaskResult {
	var out []TaskResult
	for _, t := range r.Results {
		if t.Err != nil && !t.Blocked {
			out = append(out, t)
		}
	}
	return out
}

// Blocked returns tasks that never ran because a prerequisite failed.
func (r Run) Blocked() []TaskResult {
	var out []TaskResult
	for _, t := range r.Results {
		if t.Blocked {
			out = append(out, t)
		}
	}
	return out
}

// Complete reports whether every task in the graph produced a result.
func (r Run) Complete() bool { return len(r.Failed()) == 0 && len(r.Blocked()) == 0 }

// Text returns the outputs keyed by task id, for tasks that succeeded.
func (r Run) Text() map[string]string {
	out := map[string]string{}
	for _, t := range r.Results {
		if t.Err == nil && !t.Blocked {
			out[t.Task.ID] = t.Text
		}
	}
	return out
}

// Span reports the widest concurrency the run actually reached. Compared against the stream budget
// it says whether the graph's shape, rather than the deployment, was the limit: a Span well under
// the budget on a slow run means the graph was too serial to fill the herd.
func (r Run) Span() int { return r.peakConcurrency }

// Schedule executes a graph, starting each task as soon as its own dependencies are met.
//
// Deliberately NOT a tier barrier. Tiers are the right unit for reasoning about the graph and are
// reported on the Run, but executing them as barriers makes every tier wait for its slowest member:
// a 40-second task in tier 0 holds back a tier-1 task whose actual prerequisite finished in two.
// The ordering guarantees are identical either way — a task still never starts before its
// dependencies — so the barrier buys nothing and costs the difference between the slowest task and
// the slowest relevant one. On a graph with one long pole and many short branches that difference
// is most of the wall clock.
//
// concurrency bounds how many run at once, and should be the deployment's stream budget: beyond it
// requests queue rather than parallelise, while still holding the caller's wait.
func Schedule(ctx context.Context, g Graph, ex Executor, concurrency int) (Run, error) {
	tiers, err := g.Sort()
	if err != nil {
		return Run{}, err
	}
	if concurrency < 1 {
		return Run{}, fmt.Errorf("vcontext: concurrency must be at least 1, got %d", concurrency)
	}

	tierOf := map[string]int{}
	for _, t := range tiers {
		for _, task := range t.Tasks {
			tierOf[task.ID] = t.Level
		}
	}

	dependents := map[string][]string{}
	pending := map[string]int{}
	byID := map[string]Task{}
	var order []string
	for _, t := range g.Tasks {
		byID[t.ID] = t
		order = append(order, t.ID)
		pending[t.ID] = len(t.DependsOn)
		for _, d := range t.DependsOn {
			dependents[d] = append(dependents[d], t.ID)
		}
	}

	var (
		mu       sync.Mutex
		results  = map[string]TaskResult{}
		outputs  = map[string]string{}
		inflight int
		peak     int
		wg       sync.WaitGroup
		sem      = make(chan struct{}, concurrency)
	)

	// ready is guarded by mu and holds ids whose dependencies are all satisfied.
	var ready []string
	for _, id := range order {
		if pending[id] == 0 {
			ready = append(ready, id)
		}
	}

	// blockAll marks a task and everything transitively behind it as blocked, naming the
	// original failure rather than the immediate parent — the nearest failed task is what the
	// caller has to fix.
	var blockAll func(id, cause string)
	blockAll = func(id, cause string) {
		for _, dep := range dependents[id] {
			if _, done := results[dep]; done {
				continue
			}
			results[dep] = TaskResult{
				Task: byID[dep], Tier: tierOf[dep],
				Blocked: true, BlockedBy: cause,
				Err: fmt.Errorf("vcontext: not run — prerequisite %q failed", cause),
			}
			blockAll(dep, cause)
		}
	}

	start := time.Now()
	var launch func()
	launch = func() {
		// Caller holds mu.
		for len(ready) > 0 {
			id := ready[0]
			ready = ready[1:]
			if _, done := results[id]; done {
				continue
			}
			task := byID[id]
			deps := map[string]string{}
			for _, d := range task.DependsOn {
				deps[d] = outputs[d]
			}
			inflight++
			if inflight > peak {
				peak = inflight
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				select {
				case sem <- struct{}{}:
					defer func() { <-sem }()
				case <-ctx.Done():
					mu.Lock()
					results[task.ID] = TaskResult{Task: task, Tier: tierOf[task.ID], Err: ctx.Err()}
					inflight--
					blockAll(task.ID, task.ID)
					mu.Unlock()
					return
				}
				t0 := time.Now()
				text, err := ex.Exec(ctx, task, deps)
				took := time.Since(t0)

				mu.Lock()
				defer mu.Unlock()
				inflight--
				results[task.ID] = TaskResult{
					Task: task, Text: text, Err: err, Took: took, Tier: tierOf[task.ID],
				}
				if err != nil {
					blockAll(task.ID, task.ID)
					return
				}
				outputs[task.ID] = text
				for _, dep := range dependents[task.ID] {
					pending[dep]--
					if pending[dep] == 0 {
						ready = append(ready, dep)
					}
				}
				launch()
			}()
		}
	}

	mu.Lock()
	launch()
	mu.Unlock()
	wg.Wait()

	run := Run{
		Wall:            time.Since(start),
		Tiers:           tiers,
		CriticalPath:    g.CriticalPath(),
		peakConcurrency: peak,
	}
	for _, id := range order {
		r, ok := results[id]
		if !ok {
			// Unreachable while Sort validates the graph, but a task with no result is a
			// silent hole in the answer, so it is named rather than omitted.
			r = TaskResult{Task: byID[id], Tier: tierOf[id], Blocked: true,
				Err: fmt.Errorf("vcontext: task %q was never scheduled", id)}
		}
		run.Results = append(run.Results, r)
	}
	sort.SliceStable(run.Results, func(i, j int) bool {
		return run.Results[i].Tier < run.Results[j].Tier
	})
	return run, nil
}
