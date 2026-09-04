// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

func task(id string, deps ...string) Task {
	return Task{ID: id, Desc: "do " + id, DependsOn: deps}
}

func TestSortTiersIndependentWorkTogether(t *testing.T) {
	g := Graph{Tasks: []Task{task("a"), task("b"), task("c")}}
	tiers, err := g.Sort()
	if err != nil {
		t.Fatal(err)
	}
	if len(tiers) != 1 {
		t.Fatalf("three independent tasks should form one tier, got %d", len(tiers))
	}
	if len(tiers[0].Tasks) != 3 {
		t.Fatalf("want 3 tasks in the tier, got %d", len(tiers[0].Tasks))
	}
}

func TestSortRespectsAChain(t *testing.T) {
	g := Graph{Tasks: []Task{task("c", "b"), task("b", "a"), task("a")}}
	tiers, err := g.Sort()
	if err != nil {
		t.Fatal(err)
	}
	if len(tiers) != 3 {
		t.Fatalf("a three-link chain needs three tiers, got %d", len(tiers))
	}
	for i, want := range []string{"a", "b", "c"} {
		if tiers[i].Tasks[0].ID != want {
			t.Fatalf("tier %d: want %s, got %s", i, want, tiers[i].Tasks[0].ID)
		}
	}
}

// The shape the design is for: a dependent chain interleaved with background work.
func TestSortInterleavedChainAndBackground(t *testing.T) {
	g := Graph{Tasks: []Task{
		task("summarise"), // background
		task("build", "compile"),
		task("compile", "fetch"),
		task("fetch"),
		task("lint"), // background
	}}
	tiers, err := g.Sort()
	if err != nil {
		t.Fatal(err)
	}
	if len(tiers) != 3 {
		t.Fatalf("want 3 tiers, got %d", len(tiers))
	}
	// Background work must land in tier 0 alongside the chain's head, not be serialised
	// behind it.
	var t0 []string
	for _, task := range tiers[0].Tasks {
		t0 = append(t0, task.ID)
	}
	got := strings.Join(t0, ",")
	if !strings.Contains(got, "summarise") || !strings.Contains(got, "lint") || !strings.Contains(got, "fetch") {
		t.Fatalf("tier 0 should hold both background tasks and the chain head, got %q", got)
	}
}

func TestSortNamesTheCycle(t *testing.T) {
	g := Graph{Tasks: []Task{task("a", "c"), task("b", "a"), task("c", "b")}}
	_, err := g.Sort()
	var ce *CycleError
	if !errors.As(err, &ce) {
		t.Fatalf("want a CycleError, got %v", err)
	}
	for _, id := range []string{"a", "b", "c"} {
		if !strings.Contains(err.Error(), id) {
			t.Fatalf("the error must name every task in the cycle; %q omits %s", err, id)
		}
	}
}

// A cycle deep in a graph must not be reported as if the whole graph were circular, and the
// tasks that CAN run must not be swept into it.
func TestSortCycleReportExcludesSchedulableWork(t *testing.T) {
	g := Graph{Tasks: []Task{task("ok1"), task("ok2", "ok1"), task("x", "y"), task("y", "x")}}
	_, err := g.Sort()
	var ce *CycleError
	if !errors.As(err, &ce) {
		t.Fatalf("want CycleError, got %v", err)
	}
	for _, id := range ce.Involved {
		if id == "ok1" || id == "ok2" {
			t.Fatalf("schedulable task %s reported as part of the cycle: %v", id, ce.Involved)
		}
	}
}

func TestSortRejectsAnInventedDependency(t *testing.T) {
	g := Graph{Tasks: []Task{task("a", "ghost")}}
	_, err := g.Sort()
	var me *MissingDepError
	if !errors.As(err, &me) {
		t.Fatalf("want MissingDepError, got %v", err)
	}
	if me.Missing != "ghost" {
		t.Fatalf("want the missing id named, got %q", me.Missing)
	}
}

func TestSortRejectsDuplicateIDs(t *testing.T) {
	g := Graph{Tasks: []Task{task("a"), task("a")}}
	if _, err := g.Sort(); err == nil {
		t.Fatal("duplicate ids make dependency references ambiguous and must be rejected")
	}
}

// A task must not join the same tier as the task that unblocked it.
func TestSortDependentNeverSharesATierWithItsDependency(t *testing.T) {
	g := Graph{Tasks: []Task{task("a"), task("b", "a"), task("c", "a"), task("d", "b", "c")}}
	tiers, err := g.Sort()
	if err != nil {
		t.Fatal(err)
	}
	level := map[string]int{}
	for _, tr := range tiers {
		for _, tk := range tr.Tasks {
			level[tk.ID] = tr.Level
		}
	}
	for _, tk := range g.Tasks {
		for _, d := range tk.DependsOn {
			if level[d] >= level[tk.ID] {
				t.Fatalf("%s (tier %d) depends on %s (tier %d)", tk.ID, level[tk.ID], d, level[d])
			}
		}
	}
}

func TestIndependentFindsBackgroundWork(t *testing.T) {
	g := Graph{Tasks: []Task{task("bg"), task("head"), task("tail", "head")}}
	ind := g.Independent()
	if len(ind) != 1 || ind[0].ID != "bg" {
		t.Fatalf("only bg is free of both prerequisites and dependents, got %v", ind)
	}
}

func TestCriticalPathBoundsTheRun(t *testing.T) {
	g := Graph{Tasks: []Task{task("a"), task("b", "a"), task("c", "b"), task("x")}}
	p := g.CriticalPath()
	if strings.Join(p, ">") != "a>b>c" {
		t.Fatalf("want a>b>c, got %v", p)
	}
}

// --- scheduling ---

type fakeExec struct {
	mu    sync.Mutex
	order []string
	saw   map[string]map[string]string
	delay map[string]time.Duration
	fail  map[string]bool
}

func newFake() *fakeExec {
	return &fakeExec{saw: map[string]map[string]string{}, delay: map[string]time.Duration{}, fail: map[string]bool{}}
}

func (f *fakeExec) Exec(ctx context.Context, t Task, done map[string]string) (string, error) {
	if d := f.delay[t.ID]; d > 0 {
		time.Sleep(d)
	}
	f.mu.Lock()
	f.order = append(f.order, t.ID)
	cp := map[string]string{}
	for k, v := range done {
		cp[k] = v
	}
	f.saw[t.ID] = cp
	f.mu.Unlock()
	if f.fail[t.ID] {
		return "", fmt.Errorf("boom in %s", t.ID)
	}
	return "out:" + t.ID, nil
}

func TestScheduleRunsDependenciesFirst(t *testing.T) {
	g := Graph{Tasks: []Task{task("c", "b"), task("b", "a"), task("a")}}
	f := newFake()
	run, err := Schedule(context.Background(), g, f, 8)
	if err != nil {
		t.Fatal(err)
	}
	if !run.Complete() {
		t.Fatalf("run should be complete: %+v", run.Results)
	}
	if strings.Join(f.order, ",") != "a,b,c" {
		t.Fatalf("want a,b,c got %v", f.order)
	}
}

func TestScheduleForwardsDependencyOutputs(t *testing.T) {
	g := Graph{Tasks: []Task{task("a"), task("b"), task("c", "a", "b")}}
	f := newFake()
	if _, err := Schedule(context.Background(), g, f, 8); err != nil {
		t.Fatal(err)
	}
	got := f.saw["c"]
	if got["a"] != "out:a" || got["b"] != "out:b" {
		t.Fatalf("c must receive both prerequisites' output, got %v", got)
	}
}

func TestScheduleDoesNotLeakUnrelatedOutputs(t *testing.T) {
	g := Graph{Tasks: []Task{task("a"), task("b"), task("c", "a")}}
	f := newFake()
	if _, err := Schedule(context.Background(), g, f, 8); err != nil {
		t.Fatal(err)
	}
	if _, leaked := f.saw["c"]["b"]; leaked {
		t.Fatal("c declared no dependency on b and must not receive its output — context it did not ask for is context it cannot be held to")
	}
}

// The reason this is not a tier barrier.
func TestScheduleDoesNotWaitForASlowSiblingInTheSameTier(t *testing.T) {
	// slow and fast are both tier 0; dep needs only fast.
	g := Graph{Tasks: []Task{task("slow"), task("fast"), task("dep", "fast")}}
	f := newFake()
	f.delay["slow"] = 300 * time.Millisecond
	run, err := Schedule(context.Background(), g, f, 8)
	if err != nil {
		t.Fatal(err)
	}
	if !run.Complete() {
		t.Fatal("run should be complete")
	}
	// Under a tier barrier, dep runs after slow. Under dependency-triggered dispatch it runs
	// while slow is still going.
	pos := map[string]int{}
	for i, id := range f.order {
		pos[id] = i
	}
	if pos["dep"] > pos["slow"] {
		t.Fatalf("dep waited for an unrelated slow sibling — that is the tier barrier this design avoids (order %v)", f.order)
	}
}

func TestScheduleRespectsTheConcurrencyBound(t *testing.T) {
	var tasks []Task
	for i := 0; i < 12; i++ {
		tasks = append(tasks, task(fmt.Sprintf("t%d", i)))
	}
	var mu sync.Mutex
	cur, peak := 0, 0
	ex := execFunc(func(ctx context.Context, t Task, _ map[string]string) (string, error) {
		mu.Lock()
		cur++
		if cur > peak {
			peak = cur
		}
		mu.Unlock()
		time.Sleep(20 * time.Millisecond)
		mu.Lock()
		cur--
		mu.Unlock()
		return "ok", nil
	})
	if _, err := Schedule(context.Background(), Graph{Tasks: tasks}, ex, 3); err != nil {
		t.Fatal(err)
	}
	if peak > 3 {
		t.Fatalf("concurrency bound exceeded: peak %d > 3 — beyond the stream budget requests queue while still holding the caller's wait", peak)
	}
}

func TestScheduleBlocksDependentsOfAFailure(t *testing.T) {
	g := Graph{Tasks: []Task{task("a"), task("b", "a"), task("c", "b"), task("safe")}}
	f := newFake()
	f.fail["a"] = true
	run, err := Schedule(context.Background(), g, f, 8)
	if err != nil {
		t.Fatal(err)
	}
	if run.Complete() {
		t.Fatal("a failed task must not produce a complete run")
	}
	if len(run.Failed()) != 1 || run.Failed()[0].Task.ID != "a" {
		t.Fatalf("want exactly a failed, got %v", run.Failed())
	}
	blocked := map[string]string{}
	for _, r := range run.Blocked() {
		blocked[r.Task.ID] = r.BlockedBy
	}
	if blocked["b"] != "a" || blocked["c"] != "a" {
		t.Fatalf("b and c must be blocked, naming a as the cause; got %v", blocked)
	}
	// Unrelated work still runs — a failure in one branch must not cancel the graph.
	if _, ok := run.Text()["safe"]; !ok {
		t.Fatal("an unrelated task must still run when another branch fails")
	}
}

func TestBlockedIsDistinctFromFailed(t *testing.T) {
	g := Graph{Tasks: []Task{task("a"), task("b", "a")}}
	f := newFake()
	f.fail["a"] = true
	run, _ := Schedule(context.Background(), g, f, 4)
	for _, r := range run.Results {
		if r.Task.ID == "b" && !r.Blocked {
			t.Fatal("b never ran; reporting it as a plain failure would send the caller retrying a task that cannot succeed")
		}
	}
	if len(f.order) != 1 {
		t.Fatalf("b must never be executed, but the executor saw %v", f.order)
	}
}

func TestScheduleTextOmitsFailedTasks(t *testing.T) {
	g := Graph{Tasks: []Task{task("a"), task("b")}}
	f := newFake()
	f.fail["b"] = true
	run, _ := Schedule(context.Background(), g, f, 4)
	if _, present := run.Text()["b"]; present {
		t.Fatal("a failed task must not contribute text — an answer assembled from what happened to succeed is an answer to a different question")
	}
}

func TestScheduleRejectsACyclicGraphBeforeRunningAnything(t *testing.T) {
	g := Graph{Tasks: []Task{task("a", "b"), task("b", "a")}}
	f := newFake()
	if _, err := Schedule(context.Background(), g, f, 4); err == nil {
		t.Fatal("want an error")
	}
	if len(f.order) != 0 {
		t.Fatalf("nothing may run when no valid ordering exists, but %v ran", f.order)
	}
}

func TestScheduleRejectsZeroConcurrency(t *testing.T) {
	if _, err := Schedule(context.Background(), Graph{Tasks: []Task{task("a")}}, newFake(), 0); err == nil {
		t.Fatal("zero concurrency would hang rather than fail")
	}
}

func TestScheduleReportsSpanAndCriticalPath(t *testing.T) {
	g := Graph{Tasks: []Task{task("a"), task("b"), task("c"), task("d", "a")}}
	run, err := Schedule(context.Background(), g, newFake(), 8)
	if err != nil {
		t.Fatal(err)
	}
	if run.Span() < 1 {
		t.Fatal("span should record real overlap")
	}
	if strings.Join(run.CriticalPath, ">") != "a>d" {
		t.Fatalf("want a>d, got %v", run.CriticalPath)
	}
}

func TestScheduleEmptyGraph(t *testing.T) {
	run, err := Schedule(context.Background(), Graph{}, newFake(), 4)
	if err != nil {
		t.Fatal(err)
	}
	if !run.Complete() || len(run.Results) != 0 {
		t.Fatalf("an empty graph is a complete run with no results, got %+v", run)
	}
}

func TestScheduleCancellation(t *testing.T) {
	g := Graph{Tasks: []Task{task("a"), task("b", "a")}}
	ctx, cancel := context.WithCancel(context.Background())
	ex := execFunc(func(ctx context.Context, t Task, _ map[string]string) (string, error) {
		cancel()
		<-ctx.Done()
		return "", ctx.Err()
	})
	run, err := Schedule(ctx, g, ex, 2)
	if err != nil {
		t.Fatal(err)
	}
	if run.Complete() {
		t.Fatal("a cancelled run must not report complete")
	}
}

type execFunc func(context.Context, Task, map[string]string) (string, error)

func (f execFunc) Exec(ctx context.Context, t Task, d map[string]string) (string, error) {
	return f(ctx, t, d)
}

func TestTaskGrammarShapeIsParseable(t *testing.T) {
	for _, want := range []string{"tasks", "depends_on", "id", "desc"} {
		if !strings.Contains(TaskGrammar, want) {
			t.Fatalf("the grammar must constrain %q — a field the scheduler needs but the grammar does not require is a runtime failure after GPU time is spent", want)
		}
	}
}
