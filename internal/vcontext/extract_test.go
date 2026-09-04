// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type replyRunner struct {
	out     string
	err     error
	lastReq Request
}

func (r *replyRunner) Run(ctx context.Context, req Request) (string, error) {
	r.lastReq = req
	return r.out, r.err
}

func TestExtractGraphParsesAndValidates(t *testing.T) {
	r := &replyRunner{out: `{"tasks":[
		{"id":"fetch","desc":"get the data","depends_on":[]},
		{"id":"report","desc":"write it up","depends_on":["fetch"]}]}`}
	g, err := ExtractGraph(context.Background(), r, "get data and report", 512)
	if err != nil {
		t.Fatal(err)
	}
	if len(g.Tasks) != 2 {
		t.Fatalf("got %d tasks", len(g.Tasks))
	}
	if r.lastReq.Grammar != TaskGrammar {
		t.Fatal("extraction must be grammar-constrained, or the aggregator has to parse prose")
	}
}

func TestExtractRejectsACyclicGraphBeforeReturningIt(t *testing.T) {
	r := &replyRunner{out: `{"tasks":[
		{"id":"a","desc":"x","depends_on":["b"]},
		{"id":"b","desc":"y","depends_on":["a"]}]}`}
	_, err := ExtractGraph(context.Background(), r, "req", 512)
	if err == nil {
		t.Fatal("an unsortable graph must be caught here, not after streams have run")
	}
	if !strings.Contains(err.Error(), "not executable") {
		t.Fatalf("got %v", err)
	}
}

func TestExtractRejectsAnInventedDependency(t *testing.T) {
	r := &replyRunner{out: `{"tasks":[{"id":"a","desc":"x","depends_on":["ghost"]}]}`}
	if _, err := ExtractGraph(context.Background(), r, "req", 512); err == nil {
		t.Fatal("a dependency no task provides is a broken extraction")
	}
}

func TestTruncatedResponseSaysSo(t *testing.T) {
	// The grammar constrains shape, not length: a max_tokens cut yields a valid prefix.
	r := &replyRunner{out: `{"tasks":[{"id":"a","desc":"x","depends_on":[]`}
	_, err := ExtractGraph(context.Background(), r, "req", 64)
	if err == nil || !strings.Contains(err.Error(), "truncated") {
		t.Fatalf("the likely cause should be named: %v", err)
	}
}

func TestEmptyExtractionIsAnError(t *testing.T) {
	r := &replyRunner{out: `{"tasks":[]}`}
	if _, err := ExtractGraph(context.Background(), r, "req", 512); err == nil {
		t.Fatal("no tasks means nothing to run")
	}
}

func TestExtractRequiresARequest(t *testing.T) {
	if _, err := ExtractGraph(context.Background(), &replyRunner{}, "   ", 512); err == nil {
		t.Fatal("want an error")
	}
}

func TestExtractSurfacesRunnerErrors(t *testing.T) {
	r := &replyRunner{err: errors.New("503")}
	if _, err := ExtractGraph(context.Background(), r, "req", 512); err == nil {
		t.Fatal("want the runner error")
	}
}

func TestExtractPromptAsksForDependenciesNotOrder(t *testing.T) {
	low := strings.ToLower(ExtractPrompt)
	if !strings.Contains(low, "depends_on") {
		t.Fatal("the prompt must ask what each step needs")
	}
	if strings.Contains(low, "in order") || strings.Contains(low, "ordered list") {
		t.Fatal("asking for an order makes ordering a claim with no way to check it; asking for dependencies makes it a computation")
	}
}

// --- assembly ---

func TestAssembleOrdersByTierAndNamesNothingMissing(t *testing.T) {
	g := Graph{Tasks: []Task{
		{ID: "a", Desc: "x"},
		{ID: "b", Desc: "y", DependsOn: []string{"a"}},
	}}
	run, err := Schedule(context.Background(), g, &TaskExecutor{Runner: &recRunner{reply: "TEXT"}}, 4)
	if err != nil {
		t.Fatal(err)
	}
	out := run.Assemble()
	if strings.Index(out, "## a") > strings.Index(out, "## b") {
		t.Fatal("reading order should follow dependency order")
	}
	if strings.Contains(out, "incomplete") {
		t.Fatalf("a complete run must not be labelled incomplete:\n%s", out)
	}
}

func TestAssembleNamesFailedAndBlockedWork(t *testing.T) {
	g := Graph{Tasks: []Task{
		{ID: "ok", Desc: "x"},
		{ID: "bad", Desc: "y"},
		{ID: "after", Desc: "z", DependsOn: []string{"bad"}},
	}}
	ex := execFunc(func(ctx context.Context, t Task, _ map[string]string) (string, error) {
		if t.ID == "bad" {
			return "", errors.New("engine refused")
		}
		return "TEXT", nil
	})
	run, _ := Schedule(context.Background(), g, ex, 4)
	out := run.Assemble()
	if !strings.Contains(out, "incomplete") {
		t.Fatal("an answer built from what happened to succeed is an answer to a different request")
	}
	if !strings.Contains(out, "bad did not complete") {
		t.Fatalf("the failure must be named:\n%s", out)
	}
	if !strings.Contains(out, "after was not attempted") {
		t.Fatalf("blocked work must be distinguished from failed work:\n%s", out)
	}
	if strings.Contains(out, "## bad") {
		t.Fatal("a failed task contributes no content")
	}
}
