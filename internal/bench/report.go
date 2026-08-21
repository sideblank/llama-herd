// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package bench

import (
	"encoding/json"
	"fmt"
	"io"
	"runtime"
	"strings"
	"time"
)

// Environment records everything needed to reproduce a run. A throughput figure without it
// cannot be checked by anyone, which makes it an assertion rather than a measurement.
type Environment struct {
	Timestamp   string   `json:"timestamp"`
	Version     string   `json:"llama_herd_version"`
	Commit      string   `json:"llama_herd_commit"`
	LlamaCppRef string   `json:"llama_cpp_ref"`
	GoVersion   string   `json:"go_version"`
	OS          string   `json:"os"`
	Arch        string   `json:"arch"`
	Devices     []Device `json:"devices"`

	ModelName    string `json:"model_name"`
	ModelPath    string `json:"model_path"`
	Quantization string `json:"quantization,omitempty"`
	Context      uint32 `json:"context"`
	ContextPer   uint32 `json:"context_per_stream"`
	Batch        uint32 `json:"batch"`
	MaxStreams   uint32 `json:"max_streams"`
	GPULayers    int32  `json:"gpu_layers"`
	SplitMode    string `json:"split_mode,omitempty"`
	LoadMTP      bool   `json:"load_mtp"`

	PromptTokens int `json:"prompt_tokens"`

	// LoadAvg1 is the system load at the start of the run.
	//
	// Recorded because a benchmark taken on a busy machine produces numbers that look
	// entirely plausible and are worthless. A compile running in the background is enough
	// to make single-stream throughput read fortyfold low, and nothing in the result
	// reveals it.
	LoadAvg1 float64 `json:"load_avg_1m"`
	// CPUs is the core count, so load can be read as a fraction.
	CPUs int `json:"cpus"`
}

// Busy reports whether the machine was loaded enough to distort the result.
func (e Environment) Busy() bool {
	return e.CPUs > 0 && e.LoadAvg1 > float64(e.CPUs)*0.5
}

// Device is one accelerator present during the run.
type Device struct {
	Name       string `json:"name"`
	Type       string `json:"type"`
	TotalBytes uint64 `json:"total_bytes"`
	FreeBytes  uint64 `json:"free_bytes_at_start"`
}

// Report is a full sweep: one environment, many stream counts.
type Report struct {
	Environment Environment `json:"environment"`
	Results     []*Result   `json:"results"`
}

// WriteJSON emits the machine-readable form.
func (r *Report) WriteJSON(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}

// WriteMarkdown emits a publishable summary. It states the definitions inline, so a reader
// who did not run it can tell exactly what each column means.
func (r *Report) WriteMarkdown(w io.Writer) error {
	e := r.Environment
	var b strings.Builder

	b.WriteString("# llama-herd benchmark\n\n")
	fmt.Fprintf(&b, "Run at %s with llama-herd %s (%s) against llama.cpp `%s`.\n\n",
		e.Timestamp, e.Version, shortCommit(e.Commit), e.LlamaCppRef)

	b.WriteString("## Hardware\n\n")
	if len(e.Devices) == 0 {
		b.WriteString("No dedicated-memory GPU present: this run was CPU-only.\n\n")
	} else {
		b.WriteString("| Device | Type | Memory |\n|---|---|---|\n")
		for _, d := range e.Devices {
			fmt.Fprintf(&b, "| %s | %s | %.1f GiB |\n", d.Name, d.Type, float64(d.TotalBytes)/(1<<30))
		}
		b.WriteString("\n")
	}
	fmt.Fprintf(&b, "Platform: %s/%s, Go %s. Load average at start: %.2f across %d cores.\n\n",
		e.OS, e.Arch, e.GoVersion, e.LoadAvg1, e.CPUs)

	b.WriteString("## Model\n\n")
	fmt.Fprintf(&b, "- **Name**: %s\n", e.ModelName)
	fmt.Fprintf(&b, "- **File**: `%s`\n", e.ModelPath)
	if e.Quantization != "" {
		fmt.Fprintf(&b, "- **Quantization**: %s\n", e.Quantization)
	}
	fmt.Fprintf(&b, "- **Context**: %d total, %d per stream\n", e.Context, e.ContextPer)
	fmt.Fprintf(&b, "- **Batch**: %d\n", e.Batch)
	fmt.Fprintf(&b, "- **GPU layers**: %d\n", e.GPULayers)
	if e.SplitMode != "" && e.SplitMode != "none" {
		fmt.Fprintf(&b, "- **Split mode**: %s\n", e.SplitMode)
	}
	fmt.Fprintf(&b, "- **MTP layers loaded**: %v\n", e.LoadMTP)
	fmt.Fprintf(&b, "- **Prompt**: %d tokens\n\n", e.PromptTokens)

	if e.Busy() {
		fmt.Fprintf(&b, "> **These numbers are not trustworthy.** Load average was %.1f across %d "+
			"cores when the run started, so the machine was busy with other work. A loaded "+
			"machine produces plausible-looking figures that are badly wrong. Re-run on an "+
			"idle host.\n\n", e.LoadAvg1, e.CPUs)
	}

	b.WriteString("## Results\n\n")
	b.WriteString("| Streams | Decode tok/s | End-to-end tok/s | Per-stream tok/s | TTFT p50 | TTFT p95 |\n")
	b.WriteString("|--------:|-------------:|-----------------:|-----------------:|---------:|---------:|\n")
	for _, res := range r.Results {
		fmt.Fprintf(&b, "| %d | %.1f | %.1f | %.1f | %s | %s |\n",
			res.Streams, res.DecodeTokPerSec, res.EndToEndTokPerSec, res.PerStreamTokPerSec,
			ms(res.TTFTp50), ms(res.TTFTp95))
	}

	// Speculation is reported separately because it is the lever the whole project turns
	// on, and because "loaded but contributing nothing" is a distinct, silent failure.
	anyLib := false
	for _, res := range r.Results {
		if res.Library != nil {
			anyLib = true
			break
		}
	}
	if anyLib {
		b.WriteString("\n### Speculation and prefill\n\n")
		b.WriteString("| Streams | Prefill tok/s | Decode passes | Tokens per pass | Drafted | Accepted | Acceptance | Speculation |\n")
		b.WriteString("|--------:|--------------:|--------------:|----------------:|--------:|---------:|-----------:|:------------|\n")
		for _, res := range r.Results {
			if res.Library == nil {
				continue
			}
			acc := "-"
			if res.DraftsProposed > 0 {
				acc = fmt.Sprintf("%.1f%%", res.AcceptanceRate*100)
			}
			fmt.Fprintf(&b, "| %d | %.1f | %d | %.2f | %d | %d | %s | %s |\n",
				res.Streams, res.Library.PromptTokPerSec, res.DecodePasses,
				res.TokensPerPass, res.DraftsProposed, res.DraftsAccepted, acc,
				speculationVerdict(res))
		}
		b.WriteString("\n**Acceptance** is the fraction of drafted tokens the target model kept, and it is\n")
		b.WriteString("the number that decides whether speculation earns its cost. It does not move with\n")
		b.WriteString("prompt length, stream count or host load, so it compares across runs that nothing\n")
		b.WriteString("else compares across. Drafting with near-zero acceptance is strictly worse than not\n")
		b.WriteString("drafting: the batch space is spent and the tokens are thrown away.\n\n")
		b.WriteString("One decode pass serves every active stream at once, so **tokens per pass** is\n")
		b.WriteString("normally about the number of streams running. A speculative head that lands drafts\n")
		b.WriteString("pushes it higher, so the ratio above the stream count is the acceptance rate.\n\n")
		b.WriteString("At one stream the reading is direct: 1.00 means no speculation, and anything above\n")
		b.WriteString("means drafts are being accepted. **A model whose weights carry MTP layers and still\n")
		b.WriteString("reads 1.00 is the failure worth catching** — the head is loaded, occupying VRAM, and\n")
		b.WriteString("contributing nothing.\n\n")
		b.WriteString("Passes are counted by the engine, not by the inference library, whose own counter\n")
		b.WriteString("only increments for single-token batches and so reads near zero here.\n")
	}

	b.WriteString("\n## What these numbers mean\n\n")
	b.WriteString("- **Decode tok/s** — aggregate across all streams, measured from the first token any\n")
	b.WriteString("  stream produced to the last. Prefill is excluded. This reflects the decode loop.\n")
	b.WriteString("- **End-to-end tok/s** — the same tokens over the full wall time, prefill included.\n")
	b.WriteString("  This is closer to what a user experiences and is always the lower figure.\n")
	b.WriteString("- **Per-stream tok/s** — the aggregate divided by the stream count: the share each\n")
	b.WriteString("  stream received. It falls as streams rise even while the aggregate climbs; that\n")
	b.WriteString("  trade is the point of batching.\n")
	b.WriteString("- **TTFT** — submit to first token, so it includes prefill and queueing.\n\n")
	b.WriteString("Warmup runs are discarded. Every stream uses the same prompt, so prefill cost is\n")
	b.WriteString("identical across them and the decode comparison is not skewed by prompt length.\n\n")
	b.WriteString("Reproduce with `llama-herd bench --manifest <file> --model <name>`.\n")

	_, err := io.WriteString(w, b.String())
	return err
}

// ms formats a duration for the report, keeping sub-millisecond values legible rather than
// rounding them to a misleading "0 ms".
// speculationVerdict says whether drafting ran and whether it paid.
//
// It prefers the draft counters, which mean exactly one thing, over tokens-per-pass, which
// moves with prompt length and stream mix. The ratio is the fallback for a run recorded
// before the counters existed, not the primary reading.
func speculationVerdict(r *Result) string {
	if r.DraftsProposed > 0 {
		switch {
		case r.AcceptanceRate >= 0.5:
			return fmt.Sprintf("**active**, %.0f%% accepted", r.AcceptanceRate*100)
		case r.AcceptanceRate > 0:
			return fmt.Sprintf("active but weak, %.0f%% accepted", r.AcceptanceRate*100)
		default:
			return "**drafting, nothing accepted**"
		}
	}
	if r.DecodePasses == 0 {
		return "not measured"
	}
	// Without speculation each pass yields at most one token per active stream. A margin
	// allows for passes taken while streams were finishing.
	if r.TokensPerPass > float64(r.Streams)*1.05 {
		return "**active**"
	}
	return "not active"
}

func ms(d time.Duration) string {
	switch {
	case d == 0:
		return "-"
	case d < time.Millisecond:
		return fmt.Sprintf("%.2f ms", float64(d.Nanoseconds())/1e6)
	case d < 10*time.Millisecond:
		return fmt.Sprintf("%.1f ms", float64(d.Microseconds())/1000)
	default:
		return fmt.Sprintf("%.0f ms", float64(d.Microseconds())/1000)
	}
}

func shortCommit(c string) string {
	if len(c) > 8 {
		return c[:8]
	}
	return c
}

// GoVersion is captured here so callers need not import runtime.
func GoVersion() string { return runtime.Version() }
