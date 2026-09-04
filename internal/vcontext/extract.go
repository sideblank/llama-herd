// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// ExtractPrompt is the instruction that turns a request into a task graph.
//
// It asks for dependencies rather than for an order. A model asked to "list the steps in order" will
// produce an order and no way to check it; asked what each step *needs*, it produces a graph that
// either sorts or does not, and the scheduler derives the order from that. The difference is
// whether ordering is a claim or a computation.
const ExtractPrompt = `Break the request below into tasks.

For each task give:
  id          a short lowercase identifier, unique
  desc        what to do, self-contained
  depends_on  the ids of tasks whose OUTPUT this task needs

Only list a dependency when the task genuinely needs the earlier task's result. Work that can
proceed independently must have an empty depends_on, so it can run in parallel.

Request:
`

// ExtractGraph turns a request into a validated task graph.
//
// Grammar-constrained, so the response cannot come back as prose or omit a field the scheduler
// reads. Validated with Sort before returning, so a caller never receives a graph that cannot be
// executed — a cycle or an invented dependency is reported here rather than after streams have run.
func ExtractGraph(ctx context.Context, r Runner, request string, maxTokens int) (Graph, error) {
	if r == nil {
		return Graph{}, fmt.Errorf("vcontext: ExtractGraph needs a runner")
	}
	if strings.TrimSpace(request) == "" {
		return Graph{}, fmt.Errorf("vcontext: nothing to extract from")
	}
	if maxTokens <= 0 {
		maxTokens = 2048
	}

	out, err := r.Run(ctx, Request{
		Prompt:    ExtractPrompt + request,
		Grammar:   TaskGrammar,
		MaxTokens: maxTokens,
	})
	if err != nil {
		return Graph{}, fmt.Errorf("vcontext: extracting the task graph: %w", err)
	}

	var g Graph
	if err := json.Unmarshal([]byte(out), &g); err != nil {
		// Reachable when the response was truncated mid-object: the grammar constrains shape,
		// not length, so a max_tokens cut produces valid-prefix-invalid-JSON.
		return Graph{}, fmt.Errorf("vcontext: the extracted graph did not parse (was it truncated "+
			"at %d tokens?): %w", maxTokens, err)
	}
	if len(g.Tasks) == 0 {
		return Graph{}, fmt.Errorf("vcontext: the extraction produced no tasks")
	}

	// Validate before returning. A caller that receives an unsortable graph would discover the
	// problem only once it tried to run it, having already paid for the extraction.
	if _, err := g.Sort(); err != nil {
		return Graph{}, fmt.Errorf("vcontext: the extracted graph is not executable: %w", err)
	}
	return g, nil
}

// Assemble renders a completed run as an answer.
//
// Ordered by tier, then by task id: the reading order follows the dependency order, so a result
// appears after the results it was derived from.
//
// ⛔ Failed and blocked tasks are named rather than omitted. An answer silently assembled from the
// tasks that happened to succeed is an answer to a different request, and nothing downstream can
// tell unless the gap is stated here.
func (r Run) Assemble() string {
	var b strings.Builder
	for _, t := range r.Results {
		if t.Err != nil || t.Blocked {
			continue
		}
		fmt.Fprintf(&b, "## %s\n\n%s\n\n", t.Task.ID, strings.TrimSpace(t.Text))
	}

	failed, blocked := r.Failed(), r.Blocked()
	if len(failed) == 0 && len(blocked) == 0 && !r.Cancelled() {
		return strings.TrimSpace(b.String())
	}

	b.WriteString("---\n\nThis answer is incomplete.\n")
	for _, t := range failed {
		fmt.Fprintf(&b, "- %s did not complete: %v\n", t.Task.ID, t.Err)
	}
	for _, t := range blocked {
		fmt.Fprintf(&b, "- %s was not attempted: it needed %s\n", t.Task.ID, t.BlockedBy)
	}
	return strings.TrimSpace(b.String())
}

// Cancelled reports whether any task was stopped rather than run.
func (r Run) Cancelled() bool {
	for _, t := range r.Results {
		if t.Err != nil && t.Blocked && t.BlockedBy == "" {
			return true
		}
	}
	return false
}
