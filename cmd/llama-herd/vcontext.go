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

// vcontextCmd runs the aggregate path end to end against a running engine: split an input across
// streams, extract a structured digest from each under a grammar, and merge them by code.
//
// It exists to exercise the whole path against a real engine rather than a fake runner, and to make
// the two costs visible that only appear at that scale — the straggler spread, and whether every
// chunk actually came back.
func vcontextCmd(args []string) int {
	fs := newFlagSet("vcontext")
	url := fs.String("url", "http://localhost:8080", "engine base URL")
	model := fs.String("model", "chat", "model name in the manifest")
	docPath := fs.String("doc", "", "file to process (required)")
	grammarPath := fs.String("grammar", "", "GBNF constraining each chunk's digest")
	instruction := fs.String("instruction", "Extract the system name and its port as JSON.",
		"what each chunk is asked for")
	streams := fs.Int("streams", 8, "streams the deployment can run at once")
	perStream := fs.Int("per-stream", 8874, "context tokens available to one stream")
	maxTokens := fs.Int("max-tokens", 60, "bound on each digest")
	query := fs.String("query", "",
		"a question — switches to the selective path: index, retrieve, answer once from the "+
			"passages that bear on it, rather than digesting every chunk")
	answerBudget := fs.Int("answer-budget", 0,
		"prompt tokens the answer context may spend on source (default: one stream's usable)")
	ruleSpec := fs.String("rules", "",
		"per-field merge rules, e.g. \"system=all,port=all,errors=sum,tags=union\". "+
			"Unlisted fields use collect, which reports disagreement between chunks.")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *docPath == "" {
		fmt.Fprintln(os.Stderr, "vcontext: --doc is required")
		return 2
	}

	doc, err := os.ReadFile(*docPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "vcontext:", err)
		return 1
	}
	grammar := ""
	if *grammarPath != "" {
		g, err := os.ReadFile(*grammarPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "vcontext:", err)
			return 1
		}
		grammar = string(g)
	}

	// Four characters per token for English prose. Approximate on purpose: the planner needs a
	// count, and a real tokenizer here would mean loading the model just to plan.
	count := func(s string) int { return (len(s) + 3) / 4 }

	pol := vcontext.Policy{
		PerStreamContext: *perStream,
		OutputReserve:    *perStream / 12,
		MaxChunks:        *streams,
		Shape:            vcontext.ShapeReadHeavy,
	}
	plan, err := (vcontext.Planner{Policy: pol}).Plan(count(string(doc)))
	if err != nil {
		fmt.Fprintln(os.Stderr, "vcontext:", err)
		return 1
	}
	if plan.Refused {
		fmt.Fprintln(os.Stderr, "vcontext: refused —", plan.Reason)
		return 1
	}
	fmt.Printf("plan: %s\n", plan.Reason)

	chunks, err := vcontext.Split(string(doc), plan, count)
	if err != nil {
		fmt.Fprintln(os.Stderr, "vcontext:", err)
		return 1
	}
	fmt.Printf("split: %d chunks\n", len(chunks))
	for _, c := range chunks {
		fmt.Printf("  chunk %d: %d tokens, cut on a %s boundary\n", c.Index, c.Tokens, c.Cut)
	}

	// What the spare capacity would repair, reported even though this command does not run
	// bridges — an unrepaired severed boundary is a blind spot the caller should know about.
	if budget, err := vcontext.Budget(*streams, len(chunks)); err == nil && budget.Spare > 0 {
		bridges := vcontext.PlanBridges(chunks, budget, 512, nil)
		fmt.Printf("spare: %d streams — %d boundaries worth bridging\n", budget.Spare, len(bridges))
	}

	runner := vcontext.NewHTTPRunner(*url, *model)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	if *query != "" {
		return selective(ctx, runner, chunks, string(doc), *query, pol, count, *answerBudget)
	}

	reqs := make([]vcontext.Request, 0, len(chunks))
	for _, c := range chunks {
		reqs = append(reqs, vcontext.Request{
			Chunk:     c.Index,
			Prompt:    c.Text + "\n\n" + *instruction,
			Grammar:   grammar,
			MaxTokens: *maxTokens,
		})
	}

	fmt.Printf("\ndispatching %d chunks across %d streams...\n", len(reqs), *streams)
	batch := vcontext.Dispatch(ctx, runner, reqs, *streams)

	fmt.Printf("wall %v   median %v   slowest %v   straggler ratio %.2f\n",
		batch.Wall.Round(time.Millisecond), batch.Median.Round(time.Millisecond),
		batch.Slowest.Round(time.Millisecond), batch.StragglerRatio())

	if !batch.Complete() {
		fmt.Fprintf(os.Stderr, "\n%d chunks failed:\n", len(batch.Failed))
		for _, f := range batch.Failed {
			fmt.Fprintf(os.Stderr, "  chunk %d: %v\n", f.Chunk, f.Err)
		}
		fmt.Fprintln(os.Stderr, "\nrefusing to merge — the answer would be about a document with "+
			"holes in it")
		return 1
	}

	for _, r := range batch.Results {
		fmt.Printf("  chunk %d -> %s\n", r.Chunk, strings.TrimSpace(r.Text))
	}

	digests, err := batch.Digests()
	if err != nil {
		fmt.Fprintln(os.Stderr, "\nvcontext:", err)
		return 1
	}
	rules, err := parseRules(*ruleSpec)
	if err != nil {
		fmt.Fprintln(os.Stderr, "vcontext:", err)
		return 1
	}
	merged, err := vcontext.Merge(digests, rules)
	if err != nil {
		fmt.Fprintln(os.Stderr, "\nvcontext:", err)
		return 1
	}

	fmt.Printf("\n--- merged ---\n%s", merged.Render())
	return 0
}

// parseRules reads "field=rule,field=rule".
//
// Choosing the rule is the caller's job because only they know what the fields mean. The default,
// collect, reports disagreement — which is right when chunks describe the same thing and wrong when
// each describes a different one. Measured: four sections each naming a different system were
// reported as four-way disagreement, which is true of the values and misleading about the document.
func parseRules(spec string) (map[string]vcontext.Rule, error) {
	if strings.TrimSpace(spec) == "" {
		return nil, nil
	}
	out := map[string]vcontext.Rule{}
	for _, pair := range strings.Split(spec, ",") {
		field, name, ok := strings.Cut(strings.TrimSpace(pair), "=")
		if !ok {
			return nil, fmt.Errorf("rule %q is not field=rule", pair)
		}
		switch strings.ToLower(strings.TrimSpace(name)) {
		case "collect":
			out[field] = vcontext.RuleCollect
		case "sum":
			out[field] = vcontext.RuleSum
		case "union":
			out[field] = vcontext.RuleUnion
		case "all":
			out[field] = vcontext.RuleAll
		default:
			return nil, fmt.Errorf("unknown rule %q for field %q — use collect, sum, union or all",
				name, field)
		}
	}
	return out, nil
}

// selective answers a question from the passages that bear on it, rather than digesting everything.
//
// The order matters and each step exists for a reason: the skeleton first so nothing downstream is
// reasoned about in isolation, segmentation second because locating spans is arithmetic and must
// not wait on a model, retrieval third to spend the answer budget on what was asked about, and one
// generation last so the output is a single coherent answer rather than assembled fragments.
func selective(ctx context.Context, runner vcontext.Runner, chunks []vcontext.Chunk,
	source, query string, pol vcontext.Policy, count vcontext.CountTokens, budget int) int {

	fmt.Printf("\nquery: %q\n", query)

	fmt.Print("\nbuilding skeleton... ")
	t0 := time.Now()
	skeleton, err := vcontext.BuildSkeleton(ctx, runner, chunks, source, 400, 0, count)
	if err != nil {
		fmt.Fprintln(os.Stderr, "\nvcontext:", err)
		return 1
	}
	fmt.Printf("%v\n%s\n", time.Since(t0).Round(time.Millisecond), indent(skeleton))

	ix := vcontext.SegmentAll(chunks, source, 200, count)
	ix.Skeleton = skeleton
	fmt.Printf("\nindexed %d spans covering %.0f%% of the source\n",
		len(ix.Spans), ix.Covers(source)*100)

	if budget <= 0 {
		budget = pol.PerStreamContext - pol.OutputReserve - count(skeleton) - count(query) - 64
	}
	if budget < 1 {
		fmt.Fprintln(os.Stderr, "vcontext: no budget left for source after the skeleton and query")
		return 1
	}

	sel, err := (vcontext.Selector{
		Retriever: vcontext.LexicalRetriever{},
		Count:     count,
		Budget:    budget,
	}).Select(ctx, query, ix, source)
	if err != nil {
		fmt.Fprintln(os.Stderr, "\nvcontext:", err)
		return 1
	}
	fmt.Printf("selected %d spans, %d tokens of a %d budget (%.0f%% of the source)",
		len(sel.Spans), sel.Tokens, budget, sel.Coverage*100)
	if sel.Truncated {
		fmt.Print("  [truncated: relevant spans were left out for want of budget]")
	}
	fmt.Println()
	for _, sp := range sel.Spans {
		text, _ := sp.Text(source)
		fmt.Printf("  chunk %d [%d,%d): %.70s...\n", sp.Chunk, sp.Start, sp.End,
			strings.ReplaceAll(text, "\n", " "))
	}

	assembled, err := sel.Assemble(ix, source)
	if err != nil {
		fmt.Fprintln(os.Stderr, "\nvcontext:", err)
		return 1
	}

	fmt.Print("\nanswering... ")
	t0 = time.Now()
	answer, err := runner.Run(ctx, vcontext.Request{
		Chunk:     -1,
		Prompt:    assembled + "\n\nQuestion: " + query + "\n\nAnswer from the passages above.\n\nAnswer:",
		MaxTokens: 200,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "\nvcontext:", err)
		return 1
	}
	fmt.Printf("%v\n\n--- answer ---\n%s\n", time.Since(t0).Round(time.Millisecond),
		strings.TrimSpace(answer))
	return 0
}

func indent(s string) string {
	var b strings.Builder
	for _, line := range strings.Split(s, "\n") {
		fmt.Fprintf(&b, "  %s\n", line)
	}
	return b.String()
}
