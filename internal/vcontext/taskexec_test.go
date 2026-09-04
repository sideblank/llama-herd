// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type recRunner struct {
	prompts []string
	reply   string
	err     error
}

func (r *recRunner) Run(ctx context.Context, req Request) (string, error) {
	r.prompts = append(r.prompts, req.Prompt)
	if r.err != nil {
		return "", r.err
	}
	if r.reply == "" {
		return "ok", nil
	}
	return r.reply, nil
}

func TestExecIncludesDependencyOutputsKeyedByID(t *testing.T) {
	r := &recRunner{}
	e := &TaskExecutor{Runner: r, Instruction: "Build a service."}
	task := Task{ID: "impl", Desc: "write the handler", DependsOn: []string{"types", "iface"}}
	_, err := e.Exec(context.Background(), task, map[string]string{
		"types": "type User struct{}",
		"iface": "type Repo interface{}",
	})
	if err != nil {
		t.Fatal(err)
	}
	p := r.prompts[0]
	for _, want := range []string{"--- types ---", "type User struct{}", "--- iface ---", "write the handler"} {
		if !strings.Contains(p, want) {
			t.Fatalf("prompt missing %q:\n%s", want, p)
		}
	}
}

func TestDependencyOrderIsDeterministic(t *testing.T) {
	task := Task{ID: "x", Desc: "d", DependsOn: []string{"b", "a", "c"}}
	done := map[string]string{"a": "A", "b": "B", "c": "C"}
	first := ""
	for i := 0; i < 25; i++ {
		r := &recRunner{}
		e := &TaskExecutor{Runner: r}
		if _, err := e.Exec(context.Background(), task, done); err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			first = r.prompts[0]
			continue
		}
		if r.prompts[0] != first {
			t.Fatal("Go map iteration is randomised; an unstable dependency order changes the prompt between identical runs and makes any comparison noise")
		}
	}
	if strings.Index(first, "--- a ---") > strings.Index(first, "--- b ---") {
		t.Fatal("dependencies should render in sorted order")
	}
}

func TestUndeclaredDependenciesAreNotRendered(t *testing.T) {
	r := &recRunner{}
	e := &TaskExecutor{Runner: r}
	task := Task{ID: "x", Desc: "d", DependsOn: []string{"a"}}
	// Schedule never passes an undeclared dep, but the assembler must not leak one if it did.
	_, _ = e.Exec(context.Background(), task, map[string]string{"a": "A", "sneaky": "S"})
	if strings.Contains(r.prompts[0], "sneaky") {
		t.Fatal("context a task did not ask for is context it cannot be held to")
	}
}

func TestTaskWithNoDependenciesGetsNoDependencySection(t *testing.T) {
	r := &recRunner{}
	e := &TaskExecutor{Runner: r}
	_, _ = e.Exec(context.Background(), Task{ID: "root", Desc: "start"}, nil)
	if strings.Contains(r.prompts[0], "depends on") {
		t.Fatalf("an empty dependency preamble spends budget saying nothing:\n%s", r.prompts[0])
	}
}

func TestOverBudgetPromptIsRefusedNotTruncated(t *testing.T) {
	r := &recRunner{}
	e := &TaskExecutor{Runner: r, Budget: 50, Count: CharEstimate}
	task := Task{ID: "impl", Desc: "d", DependsOn: []string{"big"}}
	_, err := e.Exec(context.Background(), task, map[string]string{"big": strings.Repeat("x ", 400)})
	var over *ErrPromptOverBudget
	if !errors.As(err, &over) {
		t.Fatalf("want ErrPromptOverBudget, got %v", err)
	}
	if over.Task != "impl" || len(over.Deps) != 1 {
		t.Fatalf("the error must name the task and its dependencies: %+v", over)
	}
	if len(r.prompts) != 0 {
		t.Fatal("nothing should be sent — truncating would silently drop part of a prerequisite")
	}
}

func TestBudgetCheckIsSkippedWithoutACounter(t *testing.T) {
	r := &recRunner{}
	// Budget set, no Count: the check cannot run, and guessing a chars-per-token divisor would
	// be confidently wrong in both directions.
	e := &TaskExecutor{Runner: r, Budget: 5}
	if _, err := e.Exec(context.Background(), Task{ID: "x", Desc: strings.Repeat("y", 500)}, nil); err != nil {
		t.Fatalf("without a token counter the budget cannot be enforced, and inventing one is worse: %v", err)
	}
}

func TestBudgetIsMeasuredOnTheAssembledPromptNotTheDescription(t *testing.T) {
	r := &recRunner{}
	e := &TaskExecutor{Runner: r, Budget: 40, Count: CharEstimate}
	// A short description whose dependencies push it over.
	_, err := e.Exec(context.Background(), Task{ID: "x", Desc: "hi", DependsOn: []string{"a"}},
		map[string]string{"a": strings.Repeat("z", 300)})
	if err == nil {
		t.Fatal("the cost is what the stream receives, not what the task declared")
	}
}

func TestRunnerErrorSurfaces(t *testing.T) {
	r := &recRunner{err: errors.New("upstream 503")}
	e := &TaskExecutor{Runner: r}
	if _, err := e.Exec(context.Background(), Task{ID: "x", Desc: "d"}, nil); err == nil {
		t.Fatal("want the runner error")
	}
}

func TestMissingRunnerIsAnError(t *testing.T) {
	e := &TaskExecutor{}
	if _, err := e.Exec(context.Background(), Task{ID: "x", Desc: "d"}, nil); err == nil {
		t.Fatal("a nil runner would panic on the first task")
	}
}

func TestSharedPrefixIsTheInstruction(t *testing.T) {
	e := &TaskExecutor{Instruction: "Build a service."}
	if !strings.HasPrefix(e.SharedPrefix(), "Build a service.") {
		t.Fatal("the shared prefix is what every stream repeats and what prefix reuse would compute once")
	}
	if (&TaskExecutor{}).SharedPrefix() != "" {
		t.Fatal("no instruction, nothing shared")
	}
}

// End to end through the scheduler.
func TestScheduleThroughTheExecutorRespectsOrderAndCarriesResults(t *testing.T) {
	r := &recRunner{reply: "RESULT"}
	e := &TaskExecutor{Runner: r, Instruction: "Do the work."}
	g := Graph{Tasks: []Task{
		{ID: "c", Desc: "third", DependsOn: []string{"b"}},
		{ID: "b", Desc: "second", DependsOn: []string{"a"}},
		{ID: "a", Desc: "first"},
	}}
	run, err := Schedule(context.Background(), g, e, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !run.Complete() {
		t.Fatalf("failed: %v blocked: %v", run.Failed(), run.Blocked())
	}
	if len(r.prompts) != 3 {
		t.Fatalf("want three requests, got %d", len(r.prompts))
	}
	// The last task must have seen its predecessor's output.
	if !strings.Contains(r.prompts[2], "RESULT") {
		t.Fatalf("task c should carry b's result:\n%s", r.prompts[2])
	}
	if !strings.Contains(r.prompts[0], "Do the work.") {
		t.Fatal("every prompt carries the shared instruction")
	}
}

func TestOverBudgetTaskBlocksItsDependentsRatherThanCorruptingThem(t *testing.T) {
	r := &recRunner{reply: strings.Repeat("w ", 300)}
	e := &TaskExecutor{Runner: r, Budget: 60, Count: CharEstimate}
	g := Graph{Tasks: []Task{
		{ID: "a", Desc: "s"},
		{ID: "b", Desc: "s", DependsOn: []string{"a"}},
		{ID: "c", Desc: "s", DependsOn: []string{"b"}},
	}}
	run, err := Schedule(context.Background(), g, e, 4)
	if err != nil {
		t.Fatal(err)
	}
	if run.Complete() {
		t.Fatal("b cannot fit a's output and must fail")
	}
	var blocked []string
	for _, r := range run.Blocked() {
		blocked = append(blocked, r.Task.ID)
	}
	if len(blocked) != 1 || blocked[0] != "c" {
		t.Fatalf("c depends on a task that never produced output, so it must be blocked rather than run against nothing; got %v", blocked)
	}
}
