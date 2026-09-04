// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/sideblank/llama-herd/internal/bench"
	"github.com/sideblank/llama-herd/internal/llama"
)

var cards = map[string]uint64{
	"3090": 24 << 30,
	"4090": 24 << 30,
	"5090": 32 << 30,
}

func fitCmd(args []string) int {
	fs := newFlagSet("fit")
	card := fs.String("card", "3090", "target card: 3090, 4090, 5090, or a size in GiB")
	streams := fs.Int("streams", 4, "streams to plan for")
	speculate := fs.Bool("speculate", false, "charge for the MTP draft context that \"type\": \"mtp\" opens")
	ctx := fs.String("context", "128k", "context per stream, e.g. 32k or 128000")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: llama-herd fit [--card 3090] [--streams 4] [--context 128k] <model.gguf>")
		return 2
	}
	path := fs.Arg(0)

	vram, ok := cards[*card]
	if !ok {
		g, err := strconv.ParseFloat(strings.TrimSuffix(*card, "g"), 64)
		if err != nil || g <= 0 {
			fmt.Fprintf(os.Stderr, "fit: unknown card %q\n", *card)
			return 2
		}
		vram = uint64(g * (1 << 30))
	}

	perStream, err := parseTokens(*ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fit:", err)
		return 2
	}

	st, err := os.Stat(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fit:", err)
		return 1
	}

	llama.Backend()
	defer llama.BackendFree()

	mp := llama.DefaultModelParams()
	mp.VocabOnly = true
	mp.NGPULayers = 0
	m, err := llama.LoadModel(path, mp)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fit: %v\n", err)
		return 1
	}
	defer m.Free()

	shape := m.Shape()
	if !shape.Valid() {
		fmt.Fprintf(os.Stderr, "fit: %s does not declare enough attention geometry "+
			"(layers=%d kv_heads=%d k_len=%d v_len=%d)\n",
			path, shape.Layers, shape.HeadsKV, shape.KeyLength, shape.ValueLength)
		return 1
	}

	mtp := m.MTP()
	in := llama.FitInput{
		Shape:       shape,
		WeightBytes: uint64(st.Size()),
		VRAMBytes:   vram,
		MTPLayers:   mtp.DeclaredLayers,
		Speculate:   *speculate,
	}

	fmt.Printf("%s\n", path)
	fmt.Printf("  %s  layers=%d  kv_heads=%d  head_dim=%d/%d  ctx_train=%d\n",
		shape.Arch, shape.Layers, shape.HeadsKV, shape.KeyLength, shape.ValueLength, shape.CtxTrain)
	if shape.Hybrid() {
		fmt.Printf("  hybrid attention: every %d%s layer caches, so %d of %d layers hold KV\n",
			shape.FullAttentionInterval, ordinal(shape.FullAttentionInterval),
			shape.KVLayers(), shape.Layers)
		fmt.Printf("  the rest use linear attention, whose state is constant-size and does not grow with context\n")
	}
	fmt.Println()
	// What was actually measured on this card, when anything was.
	//
	// A fit calculation says what will physically load; it says nothing about what runs well.
	// Printing the measured configuration beside it is the difference between "this fits" and
	// "this is the arrangement that was fastest, and here is where it stopped being faster".
	if ref, ok := bench.ReferenceFor(*card); ok {
		fmt.Printf("  measured on this card: %d streams x %d context = %.0f tok/s aggregate "+
			"(%.2fx the library's own %.0f tok/s on the same boot)\n",
			ref.Streams, ref.ContextPerStream(), ref.AggregateTokPerSec, ref.Amortisation(),
			ref.LibraryTokPerSec)
		fmt.Printf("    %s %s, llama.cpp %s", ref.Model, ref.Quant, ref.LlamaCppRef)
		if ref.ForceMMQ {
			fmt.Printf(", built with GGML_CUDA_FORCE_MMQ=ON")
		}
		fmt.Println()
		fmt.Printf("    %d streams peaked at %.0f tok/s (+%.0f%%) and %d collapsed — past the "+
			"shipped setting the gain is small and the cliff is close\n",
			ref.PeakStreams, ref.PeakTokPerSec,
			(ref.PeakTokPerSec/ref.AggregateTokPerSec-1)*100, ref.CliffStreams)
		fmt.Printf("    %s\n", ref.DepthNote)
		fmt.Println()
	}

	fmt.Printf("  card %s: %s VRAM\n", *card, llama.GiB(int64(vram)))
	fmt.Printf("  weights: %s   reserved for compute: %s\n",
		llama.GiB(st.Size()), llama.GiB(llama.DefaultOverhead))
	// Drafting from the head is a second context over the same weights, and it caches. A
	// plan that omits this loads and then dies on the first large batch, so it is stated
	// whether or not it was asked for.
	switch {
	case mtp.DeclaredLayers > 0 && *speculate:
		fmt.Printf("  speculation: charging an MTP draft context over %d declared head layer(s)\n\n",
			mtp.DeclaredLayers)
	case mtp.DeclaredLayers > 0:
		fmt.Printf("  speculation: %d MTP head layer(s) declared, NOT charged — pass --speculate\n"+
			"               to include the draft context that \"type\": \"mtp\" opens\n\n",
			mtp.DeclaredLayers)
	case *speculate:
		fmt.Printf("  speculation: requested, but this file declares no MTP layers — nothing to charge\n\n")
	default:
		fmt.Println()
	}

	fmt.Println("  Total context that fits, by KV precision:")
	fmt.Println("    precision   per token    KV budget     total context")
	precs := []llama.KVPrecision{llama.KVf16, llama.KVq8, llama.KVq5, llama.KVq4}
	for _, p := range precs {
		r := llama.Fit(in, p)
		if r.KVBudget <= 0 {
			fmt.Printf("    %-11s %-12s %-13s the weights alone do not fit\n",
				p.Name, fmt.Sprintf("%.0f KiB", r.PerToken/1024), llama.GiB(r.KVBudget))
			continue
		}
		fmt.Printf("    %-11s %-12s %-13s %s\n", p.Name,
			fmt.Sprintf("%.0f KiB", r.PerToken/1024), llama.GiB(r.KVBudget), tokens(r.TotalTokens))
	}

	fmt.Printf("\n  Asked for %d streams x %s = %s total:\n", *streams, tokens(perStream),
		tokens(int64(*streams)*perStream))
	anyFits := false
	for _, p := range precs {
		pl := llama.PlanFor(in, p, *streams, perStream)
		if pl.Fits {
			anyFits = true
			fmt.Printf("    %-11s fits\n", p.Name)
		} else {
			fmt.Printf("    %-11s short by %s\n", p.Name, llama.GiB(pl.ShortBy))
		}
	}

	fmt.Printf("\n  Weight budget left by that target (quantization moves this, not the KV):\n")
	params := m.ParamCount()
	for _, p := range precs {
		b := llama.BudgetFor(in, p, *streams, perStream, params)
		if b.Bytes <= 0 {
			fmt.Printf("    KV %-9s %-14s %s\n", p.Name, llama.GiB(b.Bytes), b.Verdict())
			continue
		}
		bits := "?"
		if b.BitsPerWeight > 0 {
			bits = fmt.Sprintf("%.2f bits/wt", b.BitsPerWeight)
		}
		fmt.Printf("    KV %-9s %-14s %-15s %s\n", p.Name, llama.GiB(b.Bytes), bits, b.Verdict())
	}

	if !anyFits {
		best := llama.Fit(in, llama.KVq4)
		fmt.Printf("\n  Not reachable on one %s. At q4_0 this card holds %s of context in total,\n",
			*card, tokens(best.TotalTokens))
		if *streams > 0 {
			fmt.Printf("  which is %s per stream across %d streams.\n",
				tokens(best.TotalTokens/int64(*streams)), *streams)
		}
		short := llama.PlanFor(in, llama.KVq4, *streams, perStream).ShortBy
		extra := (short + int64(vram) - 1) / int64(vram)
		fmt.Printf("\n  Closing the gap needs one of:\n")
		fmt.Printf("    - %d more card(s) of this size, layer-split (KV splits with the layers)\n", extra)
		fmt.Printf("    - fewer streams, or less context per stream\n")
		fmt.Printf("    - a smaller model, or a smaller quantization\n")
	}
	return 0
}

func ordinal(n int) string {
	switch n {
	case 1:
		return "st"
	case 2:
		return "nd"
	case 3:
		return "rd"
	default:
		return "th"
	}
}

func parseTokens(s string) (int64, error) {
	s = strings.TrimSpace(strings.ToLower(s))
	mult := int64(1)
	switch {
	case strings.HasSuffix(s, "k"):
		mult, s = 1024, strings.TrimSuffix(s, "k")
	case strings.HasSuffix(s, "m"):
		mult, s = 1024*1024, strings.TrimSuffix(s, "m")
	}
	n, err := strconv.ParseFloat(s, 64)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("bad context %q", s)
	}
	return int64(n * float64(mult)), nil
}

func tokens(n int64) string {
	switch {
	case n >= 1024*1024:
		return fmt.Sprintf("%.2fM tokens", float64(n)/(1024*1024))
	case n >= 1024:
		return fmt.Sprintf("%.0fk tokens", float64(n)/1024)
	default:
		return fmt.Sprintf("%d tokens", n)
	}
}
