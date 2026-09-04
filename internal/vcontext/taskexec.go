// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

// TaskExecutor runs graph tasks against a herd, assembling each task's prompt from its own
// description plus the outputs of the tasks it declared a dependency on.
//
// This is what makes Schedule usable: the scheduler owns ordering and failure propagation, and this
// owns what a task actually sees.
type TaskExecutor struct {
	// Runner sends one request. Usually an *HTTPRunner.
	Runner Runner
	// Instruction is prepended to every task, describing the job as a whole. It is the same
	// bytes on every stream, so it is a prefix-reuse candidate — see docs/PREFIX-REUSE.md.
	Instruction string
	// MaxTokens bounds each task's response.
	MaxTokens int
	// Grammar optionally constrains every task's output.
	Grammar string

	// Budget is the per-stream token budget an assembled prompt must fit. Zero disables the
	// check.
	Budget int
	// Count measures a prompt in tokens. Required for the budget check to run at all.
	//
	// Deliberately not defaulted to a character heuristic: character count is not a proxy for
	// token count — flattening JSON cuts 17.8% of characters and 0.0% of tokens — so a guessed
	// divisor would produce a budget check that is confidently wrong in both directions. A
	// caller that wants an estimate opts into one explicitly with CharEstimate.
	Count func(string) int
}

// CharEstimate is a rough tokens-from-characters approximation, offered so a caller can opt into an
// estimate knowingly.
//
// ⛔ It is an estimate and it is not reliable: the ratio moves with the script, the vocabulary and
// how much punctuation and whitespace the text carries. Use a real tokenizer where the answer
// matters. It is here so that "no budget check at all" is not the only alternative.
func CharEstimate(s string) int { return (len(s) + 3) / 4 }

// ErrPromptOverBudget is returned when an assembled prompt will not fit a stream.
type ErrPromptOverBudget struct {
	Task   string
	Tokens int
	Budget int
	Deps   []string
}

func (e *ErrPromptOverBudget) Error() string {
	return fmt.Sprintf("vcontext: task %q assembles to %d tokens against a %d-token budget "+
		"(dependencies: %s) — truncating would silently drop part of a prerequisite, so the task "+
		"is refused instead",
		e.Task, e.Tokens, e.Budget, strings.Join(e.Deps, ", "))
}

// Exec builds the prompt for one task and runs it.
//
// done carries only the outputs of tasks this one declared a dependency on; Schedule enforces that.
// They are rendered keyed by task id and in sorted order — never concatenated positionally, because
// a dependent that cannot tell which prerequisite produced which text will attribute one to the
// other, and never in map order, because that would change the prompt between identical runs and
// make any comparison noise.
func (e *TaskExecutor) Exec(ctx context.Context, t Task, done map[string]string) (string, error) {
	if e.Runner == nil {
		return "", fmt.Errorf("vcontext: TaskExecutor has no runner")
	}
	prompt, deps := e.assemble(t, done)

	if e.Budget > 0 && e.Count != nil {
		if n := e.Count(prompt); n > e.Budget {
			return "", &ErrPromptOverBudget{Task: t.ID, Tokens: n, Budget: e.Budget, Deps: deps}
		}
	}

	return e.Runner.Run(ctx, Request{
		Prompt:    prompt,
		Grammar:   e.Grammar,
		MaxTokens: e.MaxTokens,
	})
}

// assemble renders the prompt and returns it with the dependency ids it included.
func (e *TaskExecutor) assemble(t Task, done map[string]string) (string, []string) {
	var deps []string
	for _, d := range t.DependsOn {
		if _, ok := done[d]; ok {
			deps = append(deps, d)
		}
	}
	sort.Strings(deps)

	var b strings.Builder
	if e.Instruction != "" {
		b.WriteString(e.Instruction)
		b.WriteString("\n\n")
	}
	if len(deps) > 0 {
		b.WriteString("Results of the steps this one depends on:\n\n")
		for _, d := range deps {
			fmt.Fprintf(&b, "--- %s ---\n%s\n\n", d, done[d])
		}
	}
	b.WriteString("Now do this step")
	if t.ID != "" {
		fmt.Fprintf(&b, " (%s)", t.ID)
	}
	b.WriteString(":\n")
	b.WriteString(t.Desc)
	return b.String(), deps
}

// SharedPrefix returns the leading text every task's prompt begins with.
//
// Identical across all streams, so it is what a prefix-reuse pass would compute once instead of
// once per stream. Exposed so a caller can measure the duplication before deciding whether to
// enable sharing.
func (e *TaskExecutor) SharedPrefix() string {
	if e.Instruction == "" {
		return ""
	}
	return e.Instruction + "\n\n"
}

var _ Executor = (*TaskExecutor)(nil)
