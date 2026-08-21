// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package bench

import (
	"fmt"
	"strings"
)

// Workload is a synthetic request shaped like real traffic.
//
// Benchmarks usually use one generic prompt, which measures the model and hides everything
// that depends on what is being asked. Speculation is the clearest example: a drafter that
// predicts from context is worth a great deal when output repeats the prompt and nothing at
// all when it does not, so a single prompt reports one point on a curve and calls it the
// answer.
type Workload struct {
	Name string
	// Description says what real traffic this stands in for.
	Description string
	// Prompt is the request text.
	Prompt string
	// Repetitive records whether output is expected to echo the prompt. This is the
	// property that decides whether context-predicting speculation can help at all, so a
	// report that omits it is not comparable with one from different traffic.
	Repetitive bool
}

// Workloads returns the built-in set, ordered from most to least repetitive.
func Workloads() []Workload {
	return []Workload{
		{
			Name:        "code-edit",
			Description: "A file in the prompt, one small change requested. The output restates most of the input.",
			Repetitive:  true,
			Prompt:      codeEditPrompt(),
		},
		{
			Name:        "schema-fill",
			Description: "A schema in the prompt, filled with values. Structure repeats, values do not.",
			Repetitive:  true,
			Prompt:      schemaFillPrompt(),
		},
		{
			Name:        "transcript",
			Description: "A conversation continued in the same format. Speaker labels and structure repeat.",
			Repetitive:  true,
			Prompt:      transcriptPrompt(),
		},
		{
			Name:        "summarize",
			Description: "A long document reduced to a short summary. Output shares vocabulary but not structure.",
			Repetitive:  false,
			Prompt:      summarizePrompt(),
		},
		{
			Name:        "freeform",
			Description: "Open-ended prose from a short prompt. Nothing to predict from context.",
			Repetitive:  false,
			Prompt:      "Explain how memory bandwidth limits inference throughput on consumer GPUs.",
		},
	}
}

// WorkloadByName returns one workload, or an error naming the available set.
func WorkloadByName(name string) (Workload, error) {
	all := Workloads()
	for _, w := range all {
		if w.Name == name {
			return w, nil
		}
	}
	names := make([]string, len(all))
	for i, w := range all {
		names[i] = w.Name
	}
	return Workload{}, fmt.Errorf("unknown workload %q; available: %s", name, strings.Join(names, ", "))
}

func codeEditPrompt() string {
	var b strings.Builder
	b.WriteString("Here is a Go file. Add a `Timeout` field to Config and use it in Run. " +
		"Output the complete file with your change applied.\n\n```go\n")
	b.WriteString("package worker\n\nimport (\n\t\"context\"\n\t\"errors\"\n\t\"sync\"\n\t\"time\"\n)\n\n")
	b.WriteString("// Config tunes the worker pool.\ntype Config struct {\n" +
		"\tWorkers int\n\tQueueSize int\n\tRetries int\n}\n\n")
	for i := 0; i < 14; i++ {
		fmt.Fprintf(&b, `
// Task%d performs unit of work %d.
func (p *Pool) Task%d(ctx context.Context, in []byte) ([]byte, error) {
	if len(in) == 0 {
		return nil, errors.New("task%d: empty input")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	return append([]byte("task%d:"), in...), nil
}
`, i, i, i, i, i)
	}
	b.WriteString("\n// Run starts the pool.\nfunc (p *Pool) Run(ctx context.Context) error {\n" +
		"\tvar wg sync.WaitGroup\n\tfor i := 0; i < p.cfg.Workers; i++ {\n\t\twg.Add(1)\n" +
		"\t\tgo func() { defer wg.Done(); p.loop(ctx) }()\n\t}\n\twg.Wait()\n\treturn nil\n}\n```\n")
	return b.String()
}

func schemaFillPrompt() string {
	var b strings.Builder
	b.WriteString("Fill this schema for eight different products. Output one JSON object per line, " +
		"using exactly these keys in this order.\n\n")
	b.WriteString(`{"sku":"","name":"","category":"","price_cents":0,"in_stock":true,` +
		`"weight_grams":0,"dimensions":{"w":0,"h":0,"d":0},"tags":[],"supplier_id":""}` + "\n\n")
	b.WriteString("Example:\n")
	b.WriteString(`{"sku":"AB-1001","name":"Widget","category":"hardware","price_cents":1299,` +
		`"in_stock":true,"weight_grams":250,"dimensions":{"w":10,"h":4,"d":4},` +
		`"tags":["small","metal"],"supplier_id":"SUP-01"}` + "\n")
	return b.String()
}

func transcriptPrompt() string {
	var b strings.Builder
	b.WriteString("Continue this support transcript in the same format for six more exchanges.\n\n")
	for i := 1; i <= 6; i++ {
		fmt.Fprintf(&b, "CUSTOMER: I am seeing error code E%03d when saving.\n", 100+i)
		fmt.Fprintf(&b, "AGENT: Thank you. Error E%03d means the record was locked. "+
			"Please close the other tab and retry.\n", 100+i)
		fmt.Fprintf(&b, "CUSTOMER: That worked, thank you.\n")
		fmt.Fprintf(&b, "AGENT: Glad to hear it. Anything else I can help with?\n\n")
	}
	return b.String()
}

func summarizePrompt() string {
	var b strings.Builder
	b.WriteString("Summarise the following report in three sentences.\n\n")
	topics := []string{
		"memory bandwidth", "cache residency", "kernel launch overhead",
		"batch scheduling", "quantization error", "attention cost",
	}
	for i, t := range topics {
		fmt.Fprintf(&b, "Section %d: %s. Measurements over the quarter showed that %s "+
			"dominated the profile under concurrent load, with the effect growing as stream "+
			"count rose. The team investigated several mitigations and recorded the results "+
			"in the appendix.\n\n", i+1, t, t)
	}
	return b.String()
}
