// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/sideblank/llama-herd/internal/bench"
	"github.com/sideblank/llama-herd/internal/engine"
	"github.com/sideblank/llama-herd/internal/llama"
	"github.com/sideblank/llama-herd/internal/manifest"
)

func parseSweep(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil || n < 1 {
			return nil, fmt.Errorf("bad stream count %q", part)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no stream counts given")
	}
	return out, nil
}

func benchCmd(args []string) int {
	// A remote run takes a different set of flags entirely, so it is dispatched before the
	// local flag set is parsed.
	for _, a := range args {
		if a == "--url" || strings.HasPrefix(a, "--url=") {
			return remoteBench(args)
		}
	}
	fs := newFlagSet("bench")
	manifestPath := fs.String("manifest", "", "path to the model manifest (required)")
	modelName := fs.String("model", "", "model to measure (defaults to the first in the manifest)")
	sweep := fs.String("streams", "1,2,4,6,8", "comma-separated stream counts to measure")
	tokens := fs.Int("tokens", 128, "tokens to generate per stream")
	warmup := fs.Int("warmup", 16, "warmup tokens, discarded")
	promptFile := fs.String("prompt-file", "", "file whose contents are used as the prompt")
	prompt := fs.String("prompt", "Write a detailed explanation of how memory bandwidth limits inference throughput.", "prompt text")
	workload := fs.String("workload", "", "named workload shaped like real traffic; see --workload list")
	jsonOut := fs.String("json", "", "write the machine-readable report here")
	mdOut := fs.String("markdown", "", "write the publishable report here")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *manifestPath == "" {
		fmt.Fprintln(os.Stderr, "bench: --manifest is required")
		return 2
	}

	counts, err := parseSweep(*sweep)
	if err != nil {
		fmt.Fprintln(os.Stderr, "bench:", err)
		return 2
	}

	mf, err := manifest.Load(*manifestPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}

	target := mf.Models[0]
	if *modelName != "" {
		found := false
		for _, m := range mf.Models {
			if m.Name == *modelName {
				target, found = m, true
				break
			}
		}
		if !found {
			fmt.Fprintf(os.Stderr, "bench: no model %q in the manifest\n", *modelName)
			return 1
		}
	}

	text := *prompt
	if *workload != "" {
		if *workload == "list" {
			for _, w := range bench.Workloads() {
				rep := "varied"
				if w.Repetitive {
					rep = "repetitive"
				}
				fmt.Printf("  %-12s %-11s %s\n", w.Name, rep, w.Description)
			}
			return 0
		}
		w, err := bench.WorkloadByName(*workload)
		if err != nil {
			fmt.Fprintln(os.Stderr, "bench:", err)
			return 2
		}
		text = w.Prompt
		fmt.Fprintf(os.Stderr, "workload %s (%s): %s\n", w.Name,
			map[bool]string{true: "repetitive", false: "varied"}[w.Repetitive], w.Description)
	}
	if *promptFile != "" {
		b, err := os.ReadFile(*promptFile)
		if err != nil {
			fmt.Fprintln(os.Stderr, "bench:", err)
			return 1
		}
		text = string(b)
	}

	// The sweep must fit the model's configured stream count, or the extra requests
	// queue instead of running concurrently and the measurement silently becomes a
	// different one.
	for _, n := range counts {
		if uint32(n) > target.Streams {
			fmt.Fprintf(os.Stderr,
				"bench: --streams asks for %d but model %q is configured for %d; "+
					"raise \"streams\" in the manifest or the extra requests will queue "+
					"rather than run concurrently\n", n, target.Name, target.Streams)
			return 2
		}
	}

	llama.Backend()
	defer llama.BackendFree()

	fmt.Fprintf(os.Stderr, "loading %s...\n", target.Path)
	runner, err := llama.OpenRunner(runnerConfig(target))
	if err != nil {
		fmt.Fprintln(os.Stderr, "bench:", err)
		return 1
	}
	defer runner.Close()

	eng := engine.New(runner, engine.Config{})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _ = eng.Run(ctx) }()

	promptToks, _ := runner.Tokenize(text, true)

	env := bench.Environment{
		Timestamp:    time.Now().UTC().Format(time.RFC3339),
		Version:      version,
		Commit:       commit,
		LlamaCppRef:  llamaCppRef,
		GoVersion:    bench.GoVersion(),
		OS:           osName(),
		Arch:         archName(),
		ModelName:    target.Name,
		ModelPath:    filepath.Clean(target.Path),
		Context:      runner.NCtx(),
		ContextPer:   runner.NCtxSeq(),
		Batch:        target.Batch,
		MaxStreams:   runner.NSeqMax(),
		GPULayers:    target.GPULayers,
		SplitMode:    target.SplitMode,
		LoadMTP:      target.LoadMTP,
		PromptTokens: len(promptToks),
	}
	env.LoadAvg1, env.CPUs = bench.LoadAverage()
	if env.Busy() {
		fmt.Fprintf(os.Stderr,
			"\nWARNING: load average is %.1f across %d cores. This machine is busy, and a\n"+
				"benchmark taken now will produce plausible-looking numbers that are wrong.\n"+
				"The report will say so, but re-running on an idle host is better.\n\n",
			env.LoadAvg1, env.CPUs)
	}
	for _, d := range llama.Devices() {
		env.Devices = append(env.Devices, bench.Device{
			Name: d.Name, Type: d.Type.String(),
			TotalBytes: d.TotalBytes, FreeBytes: d.FreeBytes,
		})
	}

	report := &bench.Report{Environment: env}
	for _, n := range counts {
		fmt.Fprintf(os.Stderr, "measuring %d stream(s)...\n", n)
		res, err := bench.Run(ctx, eng, bench.Config{
			Model: target.Name, Prompt: text, Streams: n,
			Tokens: *tokens, Warmup: *warmup, Perf: runner,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "bench:", err)
			return 1
		}
		res.Streams_ = nil // per-stream detail is noise in a summary report
		report.Results = append(report.Results, res)
		fmt.Fprintf(os.Stderr, "  decode %.1f tok/s, end-to-end %.1f tok/s, TTFT p50 %v\n",
			res.DecodeTokPerSec, res.EndToEndTokPerSec, res.TTFTp50.Round(time.Millisecond))
		if res.DecodePasses > 0 {
			spec := ""
			if res.TokensPerPass > float64(n)*1.05 {
				spec = "  speculation ACTIVE"
			}
			fmt.Fprintf(os.Stderr, "  %.2f tokens/pass over %d passes (%d streams)%s\n",
				res.TokensPerPass, res.DecodePasses, n, spec)
		}
	}

	if err := writeReport(report, *jsonOut, *mdOut); err != nil {
		fmt.Fprintln(os.Stderr, "bench:", err)
		return 1
	}
	return 0
}

func writeReport(r *bench.Report, jsonPath, mdPath string) error {
	if jsonPath != "" {
		f, err := os.Create(jsonPath)
		if err != nil {
			return err
		}
		defer f.Close()
		if err := r.WriteJSON(f); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", jsonPath)
	}
	if mdPath != "" {
		f, err := os.Create(mdPath)
		if err != nil {
			return err
		}
		defer f.Close()
		if err := r.WriteMarkdown(f); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", mdPath)
	}
	if jsonPath == "" && mdPath == "" {
		return r.WriteMarkdown(os.Stdout)
	}
	return nil
}

// remoteBench measures a deployed instance rather than a local engine.
func remoteBench(args []string) int {
	fs := newFlagSet("bench --url")
	url := fs.String("url", "", "base URL of a running llama-herd")
	key := fs.String("api-key", "", "API key, if the endpoint requires one")
	model := fs.String("model", "", "model name to address")
	sweep := fs.String("streams", "1,2,4", "comma-separated stream counts")
	tokens := fs.Int("tokens", 128, "tokens per stream")
	warmup := fs.Int("warmup", 16, "warmup tokens, discarded")
	prompt := fs.String("prompt", "Write a detailed explanation of how memory bandwidth limits inference throughput.", "prompt text")
	workload := fs.String("workload", "", "named workload shaped like real traffic")
	// Some hosts authenticate with their own header rather than a bearer token.
	hdr := fs.String("header", "", "extra header, as Name:Value; repeat with commas")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *url == "" || *model == "" {
		fmt.Fprintln(os.Stderr, "bench: --url and --model are required for a remote run")
		return 2
	}
	counts, err := parseSweep(*sweep)
	if err != nil {
		fmt.Fprintln(os.Stderr, "bench:", err)
		return 2
	}

	if *workload != "" {
		w, err := bench.WorkloadByName(*workload)
		if err != nil {
			fmt.Fprintln(os.Stderr, "bench:", err)
			return 2
		}
		*prompt = w.Prompt
		fmt.Fprintf(os.Stderr, "workload %s (%s): %s\n", w.Name,
			map[bool]string{true: "repetitive", false: "varied"}[w.Repetitive], w.Description)
	}

	ctx := context.Background()
	headers := map[string]string{}
	for _, h := range strings.Split(*hdr, ",") {
		if name, value, ok := strings.Cut(strings.TrimSpace(h), ":"); ok {
			headers[strings.TrimSpace(name)] = strings.TrimSpace(value)
		}
	}
	r := bench.NewRemote(*url, *key, *model, headers)

	c, err := r.Counters(ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, "bench:", err)
		return 1
	}
	fmt.Fprintf(os.Stderr, "target: accelerated=%v on_gpu=%v mtp_loaded=%v gpu_free=%.1f GiB\n",
		c.Accelerated, c.OnGPU, c.MTPLoaded, float64(c.GPUFreeBytes)/(1<<30))
	if c.Warning != "" {
		fmt.Fprintf(os.Stderr, "target warning: %s\n", c.Warning)
	}
	// Client-side rates include network and whatever else the host is doing. Say so once,
	// rather than letting the numbers be read as engine throughput.
	fmt.Fprintln(os.Stderr,
		"note: rates below include network and host contention. Tokens-per-pass comes from\n"+
			"      the server's own counters and is unaffected by either.")

	for _, n := range counts {
		fmt.Fprintf(os.Stderr, "\nmeasuring %d stream(s)...\n", n)
		res, err := bench.RunRemote(ctx, r, bench.Config{
			Model: *model, Prompt: *prompt, Streams: n, Tokens: *tokens, Warmup: *warmup,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "bench:", err)
			return 1
		}
		fmt.Fprintf(os.Stderr, "  client : decode %.1f tok/s, end-to-end %.1f tok/s, TTFT p50 %v\n",
			res.DecodeTokPerSec, res.EndToEndTokPerSec, res.TTFTp50.Round(time.Millisecond))
		fmt.Fprintf(os.Stderr, "  server : %.2f tokens/pass over %d passes, %d tokens, %d evictions\n",
			res.TokensPerPass, res.DecodePasses, res.ServerTokens, res.Evictions)
		if res.MTPLoaded {
			if res.TokensPerPass > float64(n)*1.05 {
				fmt.Fprintf(os.Stderr, "  MTP    : loaded and ACCEPTING — %.2fx the %d-stream baseline\n",
					res.TokensPerPass/float64(n), n)
			} else {
				fmt.Fprintf(os.Stderr, "  MTP    : loaded but NOT accepting — occupying memory, "+
					"returning nothing\n")
			}
		}
	}
	return 0
}
