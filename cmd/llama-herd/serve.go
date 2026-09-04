// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/sideblank/llama-herd/internal/api"
	"github.com/sideblank/llama-herd/internal/bench"
	"github.com/sideblank/llama-herd/internal/draft"
	"github.com/sideblank/llama-herd/internal/engine"
	"github.com/sideblank/llama-herd/internal/llama"
	"github.com/sideblank/llama-herd/internal/manifest"
)

func splitMode(s string) llama.SplitMode {
	switch s {
	case manifest.SplitLayer:
		return llama.SplitLayer
	case manifest.SplitRow:
		return llama.SplitRow
	case manifest.SplitTensor:
		return llama.SplitTensor
	default:
		return llama.SplitNone
	}
}

// sampling starts from the defaults and applies only the fields the manifest set, so an
// omitted value keeps a sensible default while an explicit zero is honoured.
func sampling(s manifest.Sampling) llama.SamplingParams {
	p := llama.DefaultSampling()
	if s.Temperature != nil {
		p.Temperature = *s.Temperature
	}
	if s.TopK != nil {
		p.TopK = *s.TopK
	}
	if s.TopP != nil {
		p.TopP = *s.TopP
	}
	if s.MinP != nil {
		p.MinP = *s.MinP
	}
	if s.RepeatLastN != nil {
		p.RepeatLastN = *s.RepeatLastN
	}
	if s.RepeatPenalty != nil {
		p.RepeatPenalty = *s.RepeatPenalty
	}
	if s.Seed != nil {
		p.Seed = *s.Seed
	}
	return p
}

func runnerConfig(m manifest.Model) llama.RunnerConfig {
	mp := llama.DefaultModelParams()
	mp.NGPULayers = m.GPULayers
	mp.MainGPU = m.MainGPU
	mp.SplitMode = splitMode(m.SplitMode)
	mp.TensorSplit = m.TensorSplit
	mp.LoadMTP = m.LoadMTP

	cp := llama.DefaultContextParams()
	cp.NCtx = m.Context
	cp.NBatch = m.Batch
	// n_ubatch is the PHYSICAL batch — the size the compute graph and its buffers are built
	// for — while n_batch is only the largest logical submission. Tying them together builds
	// every graph for the prefill chunk size and then runs decode, which submits a handful of
	// tokens, against it.
	//
	// Left alone the library picks 512, which is what it is tuned for. Only lower it, never
	// raise it to match n_batch.
	if m.UBatch > 0 {
		cp.NUBatch = m.UBatch
	}
	cp.NSeqMax = m.Streams
	cp.KVUnified = m.KVUnified
	cp.FlashAttn = m.FlashAttention
	if t, ok := llama.ParseGGMLType(m.KVTypeK); ok {
		cp.TypeK = t
	}
	if t, ok := llama.ParseGGMLType(m.KVTypeV); ok {
		cp.TypeV = t
	}
	// Tuned to the machine unless the manifest says otherwise. The library's default is four
	// threads whatever the host, which leaves a large machine idle during CPU inference and
	// over-serves a small one whose work is mostly waiting on a GPU.
	onGPU := m.GPULayers != 0 && llama.HasGPU()
	cp.NThreads = llama.AutoThreads(onGPU)
	cp.NThreadsBatch = cp.NThreads
	if m.Threads > 0 {
		cp.NThreads = m.Threads
	}
	if m.ThreadsBatch > 0 {
		cp.NThreadsBatch = m.ThreadsBatch
	}

	// Bound the logit buffer by what will actually be asked for. Left alone the library sizes
	// it for a prefill chunk — vocabulary times four bytes times batch — which is over a
	// gigabyte of device memory on a large vocabulary, reserved for positions nothing samples.
	// A drafter reading hidden states is the exception, since prefill then really does request
	// every position.
	draft := 0
	outputEverywhere := false
	if sp := m.Speculation; sp != nil && sp.Type != "" && sp.Type != "none" {
		draft = sp.MaxDraft
		outputEverywhere = sp.Type == "mtp"
	}
	cp.NOutputsMax = llama.AutoOutputsMax(int(m.Streams), draft, outputEverywhere)

	return llama.RunnerConfig{
		ModelPath:  m.Path,
		Model:      mp,
		Context:    cp,
		Sampling:   sampling(m.Sampling),
		MMProjPath: m.MMProjPath,
		VisionGPU:  m.VisionGPU,
	}
}

func serve(args []string) int {
	fs := newFlagSet("serve")
	manifestPath := fs.String("manifest", "", "path to the model manifest (required)")
	addr := fs.String("addr", "", "listen address, overriding the manifest")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *manifestPath == "" {
		fmt.Fprintln(os.Stderr, "serve: --manifest is required")
		return 2
	}

	mf, err := manifest.Load(*manifestPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if *addr != "" {
		mf.Listen = *addr
	}

	llama.Backend()
	defer llama.BackendFree()

	if devs := llama.GPUs(); len(devs) == 0 {
		log.Printf("no dedicated-memory GPU detected — models will run on CPU")
	} else {
		for _, d := range devs {
			log.Printf("device [%d] %s: %.1f GiB free of %.1f GiB",
				d.Index, d.Name, float64(d.FreeBytes)/(1<<30), float64(d.TotalBytes)/(1<<30))
		}
	}

	reg := engine.NewRegistry()
	var runners []*llama.Runner
	// Close every runner loaded so far if a later one fails: a partial load otherwise
	// leaves weights resident with nothing able to reach them.
	defer func() {
		for _, r := range runners {
			r.Close()
		}
	}()

	for _, mm := range mf.Models {
		log.Printf("loading %s from %s", mm.Name, mm.Path)
		r, err := llama.OpenRunner(runnerConfig(mm))
		if err != nil {
			fmt.Fprintf(os.Stderr, "load %s: %v\n", mm.Name, err)
			return 1
		}
		runners = append(runners, r)

		var rend engine.Renderer
		if r.HasChatTemplate() {
			rend = r
		} else {
			log.Printf("  %s has no chat template — chat requests for it will be refused", mm.Name)
		}

		ecfg := engine.Config{MaxQueue: mm.MaxQueue, AdmitContext: mm.AdmitContext}

		// Speculation writes drafts into the cache to be checked and takes back whatever
		// the target rejected. An architecture carrying recurrent state cannot rewind, so
		// the removal silently does nothing and the next batch is refused for inconsistent
		// positions — which kills the model for every stream, not just the request that
		// speculated. Hybrid attention is exactly such an architecture, and it is what the
		// long-context models here use.
		//
		// This is measured rather than inferred, and measured before anything is served.
		canRewind := true
		needsCheckpoint := false
		if sp := mm.Speculation; sp != nil && sp.Type != "" && sp.Type != "none" {
			switch support := r.CanSeqRm(); support {
			case llama.SeqRmPartial:
				// Position alone rewinds this cache; nothing extra is needed.
			case llama.SeqRmWholeOnly:
				// The attention cache still trims by position; what cannot is the
				// recurrent and sliding-window state, which is snapshotted instead.
				needsCheckpoint = true
				log.Printf("  %s speculation: this cache supports %s removal, so drafts "+
					"are rolled back with a state checkpoint", mm.Name, support)
			default:
				canRewind = false
				log.Printf("  %s speculation: disabled — this model reports %s cache "+
					"removal, and speculation must be able to take back rejected drafts. "+
					"Serving without it.", mm.Name, support)
			}
		}

		ecfg.NeedsRewind = needsCheckpoint

		if sp := mm.Speculation; sp != nil && sp.Type == "mtp" && canRewind {
			// A model whose head was stripped in quantization cannot draft. That is a
			// configuration mistake worth naming rather than a reason to refuse service,
			// so it degrades to ordinary decoding and says so.
			spec, err := llama.OpenSpeculative(r, "draft-mtp", sp.MaxDraft)
			if err != nil {
				log.Printf("  %s speculation: mtp unavailable (%v) — serving without it. "+
					"Check that this quantization kept its prediction head.", mm.Name, err)
			} else {
				ecfg.Drafter = spec
				defer spec.Close()
				log.Printf("  %s speculation: mtp, up to %d tokens", mm.Name, spec.MaxDraft())
			}
		}
		if sp := mm.Speculation; sp != nil && sp.Type == "lookup" && canRewind {
			lk := draft.NewLookup(sp.MaxDraft)
			if sp.Pattern > 0 {
				lk.N = sp.Pattern
			}
			ecfg.Drafter = lk
			log.Printf("  %s speculation: lookup, up to %d tokens, pattern %d",
				mm.Name, lk.MaxDraft(), lk.N)
		}
		if mm.KVUnified {
			log.Printf("  %s WARNING: kv_unified is experimental and untested under load. "+
				"Admission still checks a per-stream ceiling that no longer reserves anything, "+
				"so concurrent long requests can overcommit the pool and be evicted mid-answer.",
				mm.Name)
		}
		e := engine.New(r, ecfg)
		if err := reg.Add(mm.Name, e, rend); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		log.Printf("  %s ready: %d streams, %d context (%d per stream)",
			mm.Name, r.NSeqMax(), r.NCtx(), r.NCtxSeq())
		log.Printf("  %s model: %s", mm.Name, r.Summary())
		log.Printf("  %s tuned to this host: %d cores detected, %d threads, "+
			"logit buffer bounded to %d outputs",
			mm.Name, runtime.NumCPU(), r.Threads(), r.OutputsMax())
		if r.HasVision() {
			log.Printf("  %s vision: enabled (marker %q)", mm.Name, r.MediaMarker())
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	reg.Start(ctx)

	// Measure what this deployment actually delivers, before it serves anything.
	//
	// The number that matters is not the model's speed, which can be looked up, but whether
	// THIS card with THIS stream count reaches the throughput the arrangement is supposed to
	// buy. A herd that forms and amortises nothing looks healthy in every serving metric.
	//
	// Off with LLAMA_HERD_SELFTEST=off for a deployment that cannot spare the seconds.
	// The library's own measurement, if the entrypoint took one. Published beside ours
	// because a hosted runtime may offer no way to read a container's stdout, and a
	// diagnostic that can only be logged is unavailable exactly where it is needed.
	libBench := ""
	if p := os.Getenv("LLAMA_HERD_LIBBENCH_FILE"); p != "" {
		if b, err := os.ReadFile(p); err == nil {
			libBench = strings.TrimSpace(string(b))
		}
	}

	// A configuration sweep taken before serving, for the same reason: on a host with no log
	// access, a measurement that is only printed is a measurement that was not taken.
	var sweepRaw []byte
	if p := os.Getenv("LLAMA_HERD_SWEEP_FILE"); p != "" {
		if b, err := os.ReadFile(p); err == nil && json.Valid(b) {
			sweepRaw = b
		}
	}

	selftests := map[string]bench.Selftest{}
	if os.Getenv("LLAMA_HERD_SELFTEST") != "off" {
		for _, mm := range mf.Models {
			eng, err := reg.Get(mm.Name)
			if err != nil {
				continue
			}
			streams := int(mm.Streams)
			if streams < 1 {
				streams = 1
			}
			st := bench.RunSelftest(ctx, eng, streams, 32, llamaCppRef, 0)
			selftests[mm.Name] = st
			if st.Note != "" {
				log.Printf("  %s selftest: %s", mm.Name, st.Note)
			}
			log.Printf("  %s selftest: %.1f tok/s across %d streams (%.1f per stream, "+
				"%.2f tokens/pass) in %.1fs",
				mm.Name, st.AggregateTokPerSec, st.Streams, st.PerStreamTokPerSec,
				st.TokensPerPass, st.TookSeconds)
		}
	}

	apiSrv := api.New(reg).
		WithBuild(api.BuildInfo{Version: version, Commit: commit, LlamaCppRef: llamaCppRef}).
		WithLibraryBench(libBench).
		WithSweep(sweepRaw).
		WithSamplerProfile(func() *api.SamplerProfile {
			selNs, applyNs, calls, kept := llama.SamplerTimings()
			if calls == 0 {
				return nil
			}
			return &api.SamplerProfile{
				SelectMsPerToken: float64(selNs) / float64(calls) / 1e6,
				ApplyMsPerToken:  float64(applyNs) / float64(calls) / 1e6,
				AvgCandidates:    float64(kept) / float64(calls),
				Calls:            calls,
			}
		}).
		WithDevices(func() []api.DeviceInfo {
			var out []api.DeviceInfo
			for _, d := range llama.Devices() {
				out = append(out, api.DeviceInfo{
					Index: d.Index, Name: d.Name, Type: d.Type.String(),
					TotalBytes: d.TotalBytes, FreeBytes: d.FreeBytes,
					Description: d.Description,
				})
			}
			return out
		})

	for i, mm := range mf.Models {
		r := runners[i]
		apiSrv = apiSrv.WithPlacement(mm.Name, func() api.Placement {
			p := r.PlacementInfo()
			return api.Placement{
				GPULayersRequested: p.GPULayersRequested,
				LayersTotal:        p.LayersTotal,
				OnGPU:              p.OnGPU,
				ContextTotal:       p.ContextTotal,
				ContextPerSeq:      p.ContextPerSeq,
				BatchSize:          p.BatchSize,
				KVTypeK:            p.KVTypeK,
				KVTypeV:            p.KVTypeV,
				FlashAttn:          p.FlashAttn,
				MTPLoaded:          p.MTPLoaded,
			}
		})
		if st, ok := selftests[mm.Name]; ok {
			apiSrv = apiSrv.WithSelftest(mm.Name, st)
		}
	}

	srv := &http.Server{
		Addr:    mf.Listen,
		Handler: apiSrv.Handler(),
		// No write timeout: a streaming response is long-lived by design, and a
		// deadline here would sever generations mid-flight.
		ReadHeaderTimeout: 10 * time.Second,
	}

	errc := make(chan error, 1)
	go func() {
		log.Printf("listening on %s — %d model(s): %v", mf.Listen, len(reg.Names()), reg.Names())
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errc <- err
		}
	}()

	select {
	case err := <-errc:
		fmt.Fprintln(os.Stderr, err)
		return 1
	case <-ctx.Done():
		log.Print("shutting down")
	}

	shutCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutCtx)
	reg.Wait()
	return 0
}
