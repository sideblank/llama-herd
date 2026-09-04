// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/sideblank/llama-herd/internal/bench"
	"github.com/sideblank/llama-herd/internal/draft"
	"github.com/sideblank/llama-herd/internal/engine"
	"github.com/sideblank/llama-herd/internal/llama"
	"github.com/sideblank/llama-herd/internal/manifest"
)

// sweep measures a matrix of configurations against one resident copy of the weights.
//
// The cost that dominates an experiment on rented hardware is not the measurement, it is
// getting the model onto the machine: observed here at 50 to 85 minutes per boot against a
// second and a half of actual measuring. A sweep that reloaded per configuration would spend
// well over 99% of its time on the part that does not vary between configurations.
//
// So the weights are loaded once and each configuration gets its own context. Everything that
// can only be chosen at context creation — stream count, total context, batch geometry, KV
// precision — is therefore sweepable, which is exactly the set of knobs left once the engine's
// own overhead is down to a few percent.
func sweep(args []string) int {
	fs := newFlagSet("sweep")
	manifestPath := fs.String("manifest", "", "path to the model manifest (required)")
	streamsList := fs.String("streams", "", "comma-separated stream counts, e.g. 4,6,8")
	ctxList := fs.String("context", "", "comma-separated total context sizes")
	batchList := fs.String("batch", "", "comma-separated logical batch sizes")
	ubatchList := fs.String("ubatch", "", "comma-separated physical batch sizes")
	kvList := fs.String("kv", "", "comma-separated K/V precision pairs, e.g. q8_0/q8_0,q8_0/q4_0")
	tokens := fs.Int("tokens", 64, "tokens to generate per stream")
	depths := fs.String("prompt-tokens", "", "comma-separated prompt depths to measure at, e.g. 0,4096,16384")
	resident := fs.String("resident", "", "comma-separated totals of KV resident across the herd; each configuration's depth becomes total/streams")
	specList := fs.String("spec", "", "comma-separated speculation settings to compare, e.g. none,mtp")
	reps := fs.Int("reps", 2, "repetitions per configuration")
	out := fs.String("json", "", "write results to this file as JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *manifestPath == "" {
		fmt.Fprintln(os.Stderr, "sweep: --manifest is required")
		return 2
	}

	mf, err := manifest.Load(*manifestPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if len(mf.Models) == 0 {
		fmt.Fprintln(os.Stderr, "sweep: manifest has no models")
		return 1
	}
	base := mf.Models[0]

	// The library's backends must be registered before any model load; without this the
	// loader reports only that it could not open the file.
	llama.Backend()
	defer llama.BackendFree()

	// Load the weights and nothing else.
	//
	// Deliberately NOT via OpenRunner: that also builds a context, and holding it open for the
	// duration would leave a second full KV pool resident beside every configuration being
	// measured. On a 3090 with a 425,984-token allocation that is several gigabytes of VRAM
	// the measurement should have had, and it showed up as a sweep reporting a third of the
	// throughput the same build measured while serving.
	loadStart := time.Now()
	model, err := llama.LoadModel(base.Path, runnerConfig(base).Model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "sweep: loading model: %v\n", err)
		return 1
	}
	defer model.Free()
	fmt.Printf("sweep: weights resident after %.1fs, no context held — every configuration "+
		"below gets the card to itself\n", time.Since(loadStart).Seconds())

	depthList := parseUints(*depths)
	residentList := parseUints(*resident)
	if len(depthList) == 0 && len(residentList) == 0 {
		depthList = []uint32{0} // shallow, matching the startup selftest
	}
	sort.Slice(depthList, func(i, j int) bool { return depthList[i] < depthList[j] })
	sort.Slice(residentList, func(i, j int) bool { return residentList[i] < residentList[j] })

	grid := buildGrid(base, *streamsList, *ctxList, *batchList, *ubatchList, *kvList)

	// Speculation as a sweep axis, so it is compared within one boot.
	//
	// Every cross-boot comparison this project has attempted has been swamped by which node it
	// landed on — the same configuration has measured 42 and 118 tok/s on two rented 3090s. A
	// speculation verdict taken across two boots is measuring the nodes.
	if *specList != "" {
		var next []manifest.Model
		for _, name := range strings.Split(*specList, ",") {
			name = strings.TrimSpace(name)
			if name == "" {
				continue
			}
			for _, g := range grid {
				if name == "none" {
					g.Speculation = nil
				} else {
					maxDraft := 2
					if g.Speculation != nil && g.Speculation.MaxDraft > 0 {
						maxDraft = g.Speculation.MaxDraft
					}
					g.Speculation = &manifest.Speculation{Type: name, MaxDraft: maxDraft}
				}
				next = append(next, g)
			}
		}
		if len(next) > 0 {
			grid = next
		}
	}
	fmt.Printf("sweep: %d configurations x %d depths x %d reps, %d tokens per stream\n\n",
		len(grid), len(depthList), *reps, *tokens)

	type row struct {
		Streams      uint32 `json:"streams"`
		Context      uint32 `json:"context"`
		PerStream    uint32 `json:"context_per_stream"`
		Batch        uint32 `json:"batch"`
		UBatch       uint32 `json:"ubatch"`
		KV           string `json:"kv"`
		PromptTokens int    `json:"prompt_tokens"`
		// Resident is depth x streams: how much KV the whole herd holds. Throughput tracks
		// this more closely than it tracks either factor alone.
		Resident  int     `json:"resident_tokens"`
		Aggregate float64 `json:"aggregate_tok_per_sec"`
		// Prefill is how fast the herd ingests, which is what a chunking layer runs on.
		Prefill     float64 `json:"prefill_tok_per_sec,omitempty"`
		Spec        string  `json:"speculation,omitempty"`
		Drafted     uint64  `json:"drafts_proposed,omitempty"`
		Accepted    uint64  `json:"drafts_accepted,omitempty"`
		Acceptance  float64 `json:"acceptance,omitempty"`
		FreeVRAMGiB float64 `json:"free_vram_gib,omitempty"`
		PerStrTPS   float64 `json:"per_stream_tok_per_sec"`
		Single      float64 `json:"single_stream_tok_per_sec,omitempty"`
		Amort       float64 `json:"amortisation,omitempty"`
		DecodePct   float64 `json:"decode_pct,omitempty"`
		// Low and High bound the repetitions behind Aggregate, which is their median. A wide
		// spread means the machine, not the configuration, is what varied.
		Low  float64 `json:"aggregate_low,omitempty"`
		High float64 `json:"aggregate_high,omitempty"`
		Reps int     `json:"reps,omitempty"`
		Note string  `json:"note,omitempty"`
	}
	var rows []row

	fmt.Printf("%-8s %-9s %-8s %-7s %-13s %8s %10s %10s %10s %8s\n",
		"streams", "ctx/str", "batch", "ubatch", "kv", "depth", "decode", "per-stream", "prefill", "decode%")
	fmt.Println(strings.Repeat("-", 80))

	// Once a configuration fails, every larger one will too — the grid is ordered ascending and
	// what fails is capacity. Grinding through the rest costs rented GPU time to re-learn the
	// same fact: one sweep spent over half an hour measuring 80, 88 and 96 streams after 80 had
	// already failed.
	failedAt := 0

	for _, m := range grid {
		if failedAt > 0 && m.Streams >= uint32(failedAt) {
			fmt.Printf("%-8d %-9s skipped — %d streams already failed; larger will too\n",
				m.Streams, "", failedAt)
			continue
		}
		// Depths for THIS configuration. A resident total is divided by the stream count, so
		// every configuration holds the same number of tokens across the herd and only the
		// split varies — which is the only way to tell whether throughput is governed by how
		// much cache is resident or by how many sequences it is spread over. Those two predict
		// the same thing whenever stream count and depth move together, and they have in every
		// sweep so far.
		configDepths := append([]uint32(nil), depthList...)
		for _, tot := range residentList {
			if m.Streams > 0 {
				configDepths = append(configDepths, tot/m.Streams)
			}
		}
		for _, d := range configDepths {
			depth := int(d)
			r := row{Streams: m.Streams, Context: m.Context, Batch: m.Batch, UBatch: m.UBatch,
				KV: m.KVTypeK + "/" + m.KVTypeV, PromptTokens: depth, Spec: specName(m),
				Resident: depth * int(m.Streams)}
			if m.Streams > 0 {
				r.PerStream = m.Context / m.Streams
			}

			// Report the median, not the best.
			//
			// Taking the maximum over repetitions reports the luckiest run and biases every row
			// upward by however noisy the machine is — enough that a sweep's 4-stream row came out
			// 12% above the same boot's single-shot selftest, which invites exactly the wrong
			// conclusion about which measurement to trust. Min and max are kept alongside so the
			// spread is visible rather than hidden.
			// Refuse configurations the geometry cannot hold, rather than measuring them.
			//
			// A prompt has to fit inside one sequence's share of the context, along with what
			// it will generate. Asking 24 streams sharing 425,984 tokens for a 32k prompt gives
			// each of them 17,749 to hold it in — and the sweep reported a throughput figure
			// for it anyway, which is worse than an error because it looks like data.
			if depth > 0 && r.PerStream > 0 && uint32(depth+*tokens) >= r.PerStream {
				r.Note = fmt.Sprintf("skipped: prompt %d + %d generated exceeds %d per stream",
					depth, *tokens, r.PerStream)
			}

			var runs []bench.Selftest
			for rep := 0; rep < *reps && r.Note == ""; rep++ {
				st, sp, err := measureOne(model, m, *tokens, depth)
				if err != nil {
					r.Note = err.Error()
					break
				}
				runs = append(runs, st)
				r.Drafted, r.Accepted = sp.Drafted, sp.Accepted
				r.Acceptance = sp.rate()
				r.FreeVRAMGiB = float64(sp.FreeVRAM) / (1 << 30)
			}
			// A configuration that produced no tokens must say so. Reporting 0.00 with no
			// note is indistinguishable from a configuration that ran and was simply slow —
			// which is exactly how three failed rows were published as data.
			if len(runs) == 0 && r.Note == "" {
				r.Note = "no measurement completed for this configuration"
			}
			if len(runs) > 0 {
				sort.Slice(runs, func(i, j int) bool {
					return runs[i].AggregateTokPerSec < runs[j].AggregateTokPerSec
				})
				mid := runs[len(runs)/2]
				r.Aggregate = mid.AggregateTokPerSec
				r.PerStrTPS = mid.PerStreamTokPerSec
				r.Single = mid.SingleStreamTokPerSec
				r.Amort = mid.Amortisation
				r.Prefill = mid.PromptTokPerSec
				r.Low = runs[0].AggregateTokPerSec
				r.High = runs[len(runs)-1].AggregateTokPerSec
				r.Reps = len(runs)
				if mid.Phases != nil {
					r.DecodePct = mid.Phases.DecodePct
				}
				if mid.Note != "" {
					r.Note = mid.Note
				}
				if r.Aggregate == 0 && r.Note == "" {
					r.Note = "ran but produced no tokens — treat as a failed configuration"
				}
			}
			if r.Aggregate == 0 && failedAt == 0 && m.Streams > 0 {
				failedAt = int(m.Streams)
			}
			rows = append(rows, r)
			// Write after every configuration rather than at the end.
			//
			// A sweep is the most expensive thing this binary does and the most likely to be cut
			// short — by its own timeout, by the platform replacing the instance, or by a
			// configuration that does not fit and takes the process with it. Holding the results
			// until the last one turns any of those into a total loss of an hour's work, which is
			// exactly what happened 32 minutes into one.
			writeRows(*out, rows)

			if r.Note != "" && r.Aggregate == 0 {
				fmt.Printf("%-8d %-9d %-8d %-7d %-13s %8d %10s %10s %8s  %s\n",
					m.Streams, r.PerStream, m.Batch, m.UBatch, r.KV, depth, "-", "-", "-", trunc(r.Note))
				continue
			}
			fmt.Printf("%-8d %-9d %-8d %-7d %-13s %8d %10.2f %10.2f %10.1f %7.1f%%\n",
				m.Streams, r.PerStream, m.Batch, m.UBatch, r.KV, depth, r.Aggregate, r.PerStrTPS,
				r.Prefill, r.DecodePct)
		}
	}

	// The point of the table is the winner, so say which it is rather than leaving it to be
	// eyeballed — a sweep read wrongly is worse than one not run.
	bi := -1
	for i := range rows {
		if bi < 0 || rows[i].Aggregate > rows[bi].Aggregate {
			bi = i
		}
	}
	if bi >= 0 && rows[bi].Aggregate > 0 {
		b := rows[bi]
		fmt.Printf("\nbest: %d streams x %d context, batch %d, ubatch %d, kv %s, depth %d — "+
			"%.2f tok/s aggregate (median of %d, range %.1f-%.1f)\n",
			b.Streams, b.PerStream, b.Batch, b.UBatch, b.KV, b.PromptTokens, b.Aggregate,
			b.Reps, b.Low, b.High)
	}

	if *out != "" {
		writeRows(*out, rows)
		fmt.Printf("wrote %s\n", *out)
	}
	return 0
}

// writeRows persists what has been measured so far, atomically, so a reader never sees a
// half-written file and a crash never destroys the configurations that already succeeded.
func writeRows(path string, rows any) {
	if path == "" {
		return
	}
	blob, err := json.MarshalIndent(rows, "", "  ")
	if err != nil {
		return
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, blob, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "sweep: writing %s: %v\n", tmp, err)
		return
	}
	if err := os.Rename(tmp, path); err != nil {
		fmt.Fprintf(os.Stderr, "sweep: renaming %s: %v\n", tmp, err)
	}
}

func trunc(s string) string {
	if len(s) > 60 {
		return s[:57] + "..."
	}
	return s
}

// measureOne builds a context for one configuration, measures it, and tears it down.
//
// The engine is started and stopped around each configuration because a herd's throughput is a
// property of the context it was given, and reusing an engine across configurations would carry
// the previous one's cache state into the next one's numbers.
// specStats is what a speculation run has to report alongside throughput.
//
// Drafted and Accepted answer different questions and both are needed: zero drafted means no
// drafter ran at all — for mtp, almost always a quantization that dropped the head — while
// drafted with near-zero acceptance is the expensive failure, spending batch space and throwing
// every token away. Throughput alone cannot tell those apart, and they have opposite fixes.
type specStats struct {
	Drafted  uint64
	Accepted uint64
	// FreeVRAM is device headroom with this configuration's context built. Zero when there is
	// no GPU. Read at build time rather than after the sweep, because afterwards it describes
	// the serving context and reads identically for every row.
	FreeVRAM uint64
}

func (s specStats) rate() float64 {
	if s.Drafted == 0 {
		return 0
	}
	return float64(s.Accepted) / float64(s.Drafted)
}

func measureOne(model *llama.Model, m manifest.Model, tokens, promptTokens int) (bench.Selftest, specStats, error) {
	cfg := runnerConfig(m)
	r, err := llama.OpenRunnerWithModel(model, cfg)
	if err != nil {
		return bench.Selftest{}, specStats{}, err
	}
	defer r.Close()
	// Headroom with THIS configuration's context built, which is the only moment it means
	// anything. Read after the sweep it reports the serving context instead, identical for
	// every row — which is what it did, while a VRAM hypothesis went untested beside it.
	freeVRAM := freeDeviceBytes()

	// Mirror how serving builds the engine, including the drafter. A sweep that measured
	// speculation without actually attaching one would report it as free and useless at the
	// same time.
	ecfg := engine.Config{AdmitContext: m.AdmitContext}
	var spec *llama.Speculative
	if sp := m.Speculation; sp != nil && sp.Type != "" && sp.Type != "none" {
		// Drafts are written into the cache to be checked and taken back when rejected. A
		// cache that cannot rewind by position needs a state checkpoint instead, and one that
		// can do neither cannot speculate at all — attempting it there breaks every stream,
		// not just the speculating one.
		switch r.CanSeqRm() {
		case llama.SeqRmPartial:
		case llama.SeqRmWholeOnly:
			ecfg.NeedsRewind = true
		default:
			sp = nil
		}
		if sp != nil && sp.Type == "mtp" {
			sv, err := llama.OpenSpeculative(r, "draft-mtp", sp.MaxDraft)
			if err != nil {
				return bench.Selftest{}, specStats{}, fmt.Errorf("mtp unavailable: %w", err)
			}
			spec, ecfg.Drafter = sv, sv
		}
		if sp != nil && sp.Type == "lookup" {
			lk := draft.NewLookup(sp.MaxDraft)
			if sp.Pattern > 0 {
				lk.N = sp.Pattern
			}
			ecfg.Drafter = lk
		}
	}
	if spec != nil {
		defer spec.Close()
	}
	eng := engine.New(r, ecfg)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _ = eng.Run(ctx) }()

	if promptTokens <= 0 {
		// Scale the budget with the work, or the measurement caps itself.
		//
		// The default budget is a fixed 20s covering BOTH the aggregate run and the
		// single-stream reference. At 56 streams x 64 tokens that is 3,584 tokens plus warmup
		// plus the reference — enough to crowd the budget and have the aggregate computed over
		// a truncated run. Three different stream counts then report the same ~370 tok/s,
		// which reads as a plateau in the hardware and is actually the clock.
		budget := time.Duration(int(m.Streams)*tokens)*20*time.Millisecond + 30*time.Second
		st := bench.RunSelftest(ctx, eng, int(m.Streams), tokens, "", budget)
		return st, withVRAM(drafts(eng), freeVRAM), nil
	}

	// At depth, measure directly rather than through the selftest, which is fixed to a short
	// prompt on purpose.
	//
	// Generate enough tokens that decode outweighs the prefill sharing its window. The
	// aggregate decode figure opens at the first token any stream produces and closes at the
	// last, so while the other streams are still prefilling, their prefill sits inside it. With
	// a short generation and a deep prompt that window is almost entirely prefill, which reads
	// as throughput collapsing by an order of magnitude when nothing of the sort happened.
	//
	// This is the axis that decides whether a configuration is real. Decode reads the KV cache
	// for everything already in the sequence, so a herd measured with a nearly empty cache
	// reports a ceiling it will not hold at the context the product actually promises. A
	// throughput figure without its depth is only meaningful at depth zero.
	res, err := bench.Run(ctx, eng, bench.Config{
		Prompt:  fillerPrompt(promptTokens),
		Streams: int(m.Streams),
		Tokens:  depthTokens(tokens, promptTokens),
		// One warmup, not the shallow path's two. Each warmup at depth prefills the full
		// prompt on every stream — 24 streams at 32k is three quarters of a million tokens of
		// prefill per pass — so a second one buys cache warmth that is already there and costs
		// more than the measurement it protects.
		Warmup: 1,
	})
	if err != nil {
		return bench.Selftest{}, specStats{}, err
	}
	st := bench.Selftest{
		Streams:            int(m.Streams),
		AggregateTokPerSec: res.DecodeTokPerSec,
		PerStreamTokPerSec: res.PerStreamTokPerSec,
		TokensPerPass:      res.TokensPerPass,
		GenTokens:          tokens,
	}
	// Prefill throughput: every stream's prompt, over the time until the last of them has
	// finished ingesting it.
	//
	// This is the figure a chunking layer above the engine actually runs on. Splitting a large
	// input across streams pushes far more tokens in than it pulls out, so how fast the herd
	// ingests decides the job — and it is a different quantity from decode, measured over a
	// different window, free to move in its own direction with stream count.
	if res.TTFTMax > 0 {
		st.PromptTokPerSec = float64(int(m.Streams)*promptTokens) / res.TTFTMax.Seconds()
	}
	return st, drafts(eng), nil
}

// specName labels a configuration's speculation setting for reporting.
func specName(m manifest.Model) string {
	if m.Speculation == nil || m.Speculation.Type == "" {
		return "none"
	}
	return m.Speculation.Type
}

// freeDeviceBytes reports headroom on the first GPU, or 0 when there is none.
//
// Read while a configuration's context is built, which is the only moment it means anything: read
// after the sweep it describes the serving context and comes back identical for every row, which
// is exactly how a VRAM hypothesis sat untested beside six configurations.
func freeDeviceBytes() uint64 {
	for _, d := range llama.Devices() {
		if d.IsGPU() {
			return d.FreeBytes
		}
	}
	return 0
}

// withVRAM attaches the headroom reading to a configuration's result.
func withVRAM(s specStats, free uint64) specStats { s.FreeVRAM = free; return s }

// drafts reads what speculation actually did during a run.
func drafts(eng *engine.Engine) specStats {
	st := eng.Stats()
	return specStats{Drafted: st.DraftsProposed, Accepted: st.DraftsAccepted}
}

// depthTokens scales the generation length with the prompt, so decode outweighs the prefill
// that shares its measurement window. Capped, because prefill grows with depth and the sweep
// still has to finish inside its budget.
func depthTokens(base, promptTokens int) int {
	n := base + promptTokens/16
	if n > 512 {
		n = 512
	}
	return n
}

// fillerPrompt builds a prompt of roughly the requested token count.
//
// Roughly is enough: the point is to put a comparable amount into the cache before decoding, not
// to hit an exact length, and the same filler is used for every configuration so the comparison
// between them is clean. Prose rather than a repeated token, because a degenerate prompt can be
// compressed by the sampler's penalty window in ways real traffic is not.
func fillerPrompt(want int) string {
	const para = "Ocean currents redistribute heat from the equator toward the poles, and the " +
		"pattern of that transport sets regional climate far from where the water started. " +
		"Wind stress drives the surface layer, density differences drive the deep return, and " +
		"the two together close a circulation that takes centuries to complete. "
	// Roughly four characters per token for English prose.
	var b strings.Builder
	for b.Len() < want*4 {
		b.WriteString(para)
	}
	b.WriteString("\n\nSummarise the passage above in detail.")
	return b.String()
}

// buildGrid expands the comma-separated axes into concrete configurations, leaving any axis the
// caller did not name at the manifest's value.
//
// Total context is held constant per stream where the caller swept streams without naming
// contexts, because comparing 4 streams at 104k against 8 at 104k changes two things at once and
// the KV budget will refuse the second anyway.
func buildGrid(base manifest.Model, streams, ctxs, batches, ubatches, kvs string) []manifest.Model {
	grid := []manifest.Model{base}

	// Ascending, so the configurations most likely to fit are measured first. A sweep that
	// dies on its largest configuration should still have produced everything below it.
	if v := parseUints(streams); len(v) > 0 {
		sort.Slice(v, func(i, j int) bool { return v[i] < v[j] })
		var next []manifest.Model
		for _, s := range v {
			for _, g := range grid {
				g.Streams = s
				// Hold the total context: more streams then means less each, which is the
				// trade being measured rather than an accidental second variable.
				if g.AdmitContext > 0 && g.Streams > 0 && g.Context/g.Streams < g.AdmitContext {
					g.AdmitContext = g.Context / g.Streams
				}
				next = append(next, g)
			}
		}
		grid = next
	}
	grid = expand(grid, parseUints(ctxs), func(g *manifest.Model, v uint32) {
		g.Context = v
		if g.Streams > 0 && g.AdmitContext > v/g.Streams {
			g.AdmitContext = v / g.Streams
		}
	})
	grid = expand(grid, parseUints(batches), func(g *manifest.Model, v uint32) { g.Batch = v })
	grid = expand(grid, parseUints(ubatches), func(g *manifest.Model, v uint32) { g.UBatch = v })

	if kvs != "" {
		var next []manifest.Model
		for _, pair := range strings.Split(kvs, ",") {
			k, v, ok := strings.Cut(strings.TrimSpace(pair), "/")
			if !ok {
				continue
			}
			for _, g := range grid {
				g.KVTypeK, g.KVTypeV = k, v
				next = append(next, g)
			}
		}
		if len(next) > 0 {
			grid = next
		}
	}
	return grid
}

func expand(grid []manifest.Model, vals []uint32, set func(*manifest.Model, uint32)) []manifest.Model {
	if len(vals) == 0 {
		return grid
	}
	var next []manifest.Model
	for _, v := range vals {
		for _, g := range grid {
			set(&g, v)
			next = append(next, g)
		}
	}
	return next
}

func parseUints(s string) []uint32 {
	var out []uint32
	for _, f := range strings.Split(s, ",") {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		n, err := strconv.ParseUint(f, 10, 32)
		if err != nil {
			continue
		}
		out = append(out, uint32(n))
	}
	return out
}
