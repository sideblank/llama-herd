// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/sideblank/llama-herd/internal/api"
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
	cp.NUBatch = m.Batch
	cp.NSeqMax = m.Streams
	cp.KVUnified = m.KVUnified
	cp.FlashAttn = m.FlashAttention
	if t, ok := llama.ParseGGMLType(m.KVTypeK); ok {
		cp.TypeK = t
	}
	if t, ok := llama.ParseGGMLType(m.KVTypeV); ok {
		cp.TypeV = t
	}
	if m.Threads > 0 {
		cp.NThreads = m.Threads
	}
	if m.ThreadsBatch > 0 {
		cp.NThreadsBatch = m.ThreadsBatch
	}

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

		ecfg := engine.Config{MaxQueue: mm.MaxQueue}
		if sp := mm.Speculation; sp != nil && sp.Type == "lookup" {
			lk := draft.NewLookup(sp.MaxDraft)
			if sp.Pattern > 0 {
				lk.N = sp.Pattern
			}
			ecfg.Drafter = lk
			log.Printf("  %s speculation: lookup, up to %d tokens, pattern %d",
				mm.Name, lk.MaxDraft(), lk.N)
		}
		e := engine.New(r, ecfg)
		if err := reg.Add(mm.Name, e, rend); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		log.Printf("  %s ready: %d streams, %d context (%d per stream)",
			mm.Name, r.NSeqMax(), r.NCtx(), r.NCtxSeq())
		log.Printf("  %s model: %s", mm.Name, r.Summary())
		if r.HasVision() {
			log.Printf("  %s vision: enabled (marker %q)", mm.Name, r.MediaMarker())
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	reg.Start(ctx)

	apiSrv := api.New(reg).
		WithBuild(api.BuildInfo{Version: version, Commit: commit, LlamaCppRef: llamaCppRef}).
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
