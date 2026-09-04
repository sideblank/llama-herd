// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/sideblank/llama-herd/internal/vcontext"
)

// tasksCmd runs the DAG path end to end against a running engine: extract a task graph from a
// request, execute it in dependency order, and assemble the answer.
//
// It exists to exercise the path against a real engine rather than a fake executor, and to make
// visible the two things only a real run shows — whether the extracted graph is any good, and
// whether the parallelism the graph allows is parallelism the deployment can use.
func tasksCmd(args []string) int {
	fs := newFlagSet("tasks")
	url := fs.String("url", "http://localhost:8080", "engine base URL")
	model := fs.String("model", "chat", "model name in the manifest")
	request := fs.String("request", "", "the request to decompose (or use --file)")
	file := fs.String("file", "", "read the request from a file")
	instruction := fs.String("instruction", "", "prepended to every task's prompt")
	streams := fs.Int("streams", 8, "tasks to run at once")
	maxTokens := fs.Int("max-tokens", 600, "bound on each task's answer")
	extractTokens := fs.Int("extract-tokens", 2048, "bound on the graph extraction")
	perStream := fs.Int("per-stream", 8874, "context tokens available to one stream")
	dryRun := fs.Bool("dry-run", false, "extract and print the plan without executing it")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	text := *request
	if *file != "" {
		b, err := os.ReadFile(*file)
		if err != nil {
			fmt.Fprintln(os.Stderr, "tasks:", err)
			return 1
		}
		text = string(b)
	}
	if strings.TrimSpace(text) == "" {
		fmt.Fprintln(os.Stderr, "tasks: --request or --file is required")
		return 2
	}

	runner := vcontext.NewHTTPRunner(*url, *model)
	ctx := context.Background()

	t0 := time.Now()
	g, err := vcontext.ExtractGraph(ctx, runner, text, *extractTokens)
	if err != nil {
		fmt.Fprintln(os.Stderr, "tasks:", err)
		return 1
	}
	extractWall := time.Since(t0)

	tiers, err := g.Sort()
	if err != nil {
		fmt.Fprintln(os.Stderr, "tasks:", err)
		return 1
	}

	fmt.Printf("extracted %d tasks in %s\n\n", len(g.Tasks), extractWall.Round(time.Millisecond))
	for _, tier := range tiers {
		var ids []string
		for _, t := range tier.Tasks {
			ids = append(ids, t.ID)
		}
		fmt.Printf("  tier %d  %s\n", tier.Level, strings.Join(ids, ", "))
	}
	// The critical path bounds what any number of streams could achieve. Printed before the run
	// because a request too serial to fill the herd is a property of the request, and knowing it
	// beforehand beats inferring it from a disappointing wall clock.
	cp := g.CriticalPath()
	fmt.Printf("\n  critical path : %s (%d of %d tasks)\n", strings.Join(cp, " -> "), len(cp), len(g.Tasks))
	widest := 0
	for _, tier := range tiers {
		if len(tier.Tasks) > widest {
			widest = len(tier.Tasks)
		}
	}
	fmt.Printf("  widest tier   : %d (streams available: %d)\n", widest, *streams)
	if widest < *streams {
		fmt.Printf("  ↳ the graph, not the deployment, is the limit here\n")
	}

	if *dryRun {
		return 0
	}

	ex := &vcontext.TaskExecutor{
		Runner:      runner,
		Instruction: *instruction,
		MaxTokens:   *maxTokens,
		Budget:      *perStream,
		// An estimate, and labelled as one: a real tokenizer would need the model loaded here,
		// which this command deliberately does not do. It catches an order-of-magnitude
		// overflow, not a marginal one.
		Count: vcontext.CharEstimate,
	}

	fmt.Printf("\nrunning %d tasks, %d at a time...\n\n", len(g.Tasks), *streams)
	t1 := time.Now()
	run, err := vcontext.Schedule(ctx, g, ex, *streams)
	if err != nil {
		fmt.Fprintln(os.Stderr, "tasks:", err)
		return 1
	}

	fmt.Println(run.Assemble())

	// Span against the stream budget answers a question the wall clock alone cannot: whether the
	// herd was underused because the graph was serial or because the deployment was small.
	fmt.Printf("\n---\nwall %s (extract %s, run %s) · peak concurrency %d/%d\n",
		(extractWall + time.Since(t1)).Round(time.Millisecond),
		extractWall.Round(time.Millisecond),
		time.Since(t1).Round(time.Millisecond),
		run.Span(), *streams)

	if !run.Complete() {
		return 1
	}
	return 0
}
