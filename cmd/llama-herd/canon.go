// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/sideblank/llama-herd/internal/llama"
	"github.com/sideblank/llama-herd/internal/vcontext"
)

// canonCmd measures each canonicalisation pass against a real tokenizer.
//
// This is the validation rule rather than an estimate. A pass that removes characters can still
// INCREASE tokens: BPE vocabularies carry single tokens for a leading space with a word and for a
// double newline, so a transformation that breaks one of those merges makes the tokenizer fall back
// to fragments. Whether a pass helps depends on the tokenizer and on the text, and the only way to
// know is to tokenise both sides.
//
// A pass that costs tokens on your corpus should be dropped for your corpus. The table it prints is
// the evidence for that decision.
func canonCmd(args []string) int {
	fs := newFlagSet("canon")
	modelPath := fs.String("model", "", "GGUF whose tokenizer to measure with (required)")
	docPath := fs.String("doc", "", "text file to measure (required)")
	asJSON := fs.Bool("json", false, "treat the document as a JSON array of records and measure the flattened forms")
	write := fs.String("out", "", "write the canonicalised text here")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *modelPath == "" || *docPath == "" {
		fmt.Fprintln(os.Stderr, "canon: --model and --doc are required")
		return 2
	}

	raw, err := os.ReadFile(*docPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "canon:", err)
		return 1
	}

	llama.Backend()
	defer llama.BackendFree()

	// Vocabulary only: tokenising needs no weights, and loading them would make a text
	// measurement wait on a multi-gigabyte read.
	mp := llama.DefaultModelParams()
	mp.VocabOnly = true
	mp.NGPULayers = 0
	m, err := llama.LoadModel(*modelPath, mp)
	if err != nil {
		fmt.Fprintf(os.Stderr, "canon: %v\n", err)
		return 1
	}
	defer m.Free()
	vocab := m.Vocab()

	count := func(s string) int {
		toks, err := vocab.Tokenize(s, false, false)
		if err != nil {
			return 0
		}
		return len(toks)
	}

	if *asJSON {
		recs, err := vcontext.FlattenJSONArray(strings.NewReader(string(raw)), vcontext.DefaultJSONCanon())
		if err != nil {
			fmt.Fprintln(os.Stderr, "canon:", err)
			return 1
		}
		jr := vcontext.MeasureJSONCanon(string(raw), recs, count)
		base := jr[0]
		fmt.Printf("%-20s %10s %10s %9s %9s\n", "form", "chars", "tokens", "char-cut", "tok-cut")
		for _, r := range jr {
			cc := 100 * float64(base.Chars-r.Chars) / float64(base.Chars)
			tc := 100 * float64(base.Tokens-r.Tokens) / float64(base.Tokens)
			fmt.Printf("%-20s %10d %10d %8.1f%% %8.1f%%\n", r.Name, r.Chars, r.Tokens, cc, tc)
		}
		fmt.Printf("\nrecords: %d\n", len(recs))
		return 0
	}

	results := vcontext.MeasureCanon(string(raw), count)
	if len(results) == 0 {
		fmt.Fprintln(os.Stderr, "canon: nothing measured")
		return 1
	}

	first, last := results[0], results[len(results)-1]
	fmt.Printf("%s\n%d chars, %d tokens\n\n", *docPath, first.BeforeChars, first.BeforeToks)
	fmt.Printf("%-24s %10s %10s %9s   %s\n", "pass", "chars", "tokens", "saved", "verdict")
	fmt.Println("--------------------------------------------------------------------------")

	for _, r := range results {
		verdict := "keep"
		if !r.Helped() {
			if r.Saved() == 0 {
				verdict = "no effect on this text"
			} else {
				verdict = "DROP — broke a merge and cost tokens"
			}
		}
		fmt.Printf("%-24s %10d %10d %9d   %s\n",
			r.Name, r.AfterChars, r.AfterToks, r.Saved(), verdict)
	}

	savedTok := first.BeforeToks - last.AfterToks
	pct := 0.0
	if first.BeforeToks > 0 {
		pct = float64(savedTok) / float64(first.BeforeToks) * 100
	}
	fmt.Printf("\ntotal: %d -> %d tokens (%.1f%%), %d saved\n",
		first.BeforeToks, last.AfterToks, pct, savedTok)

	// What that is worth, in the only unit that matters here.
	const prefillTokPerSec = 3300
	fmt.Printf("at ~%d tok/s prefill that is %.2fs off this document; scaled to 256k tokens, "+
		"%.1fs\n", prefillTokPerSec, float64(savedTok)/prefillTokPerSec,
		pct/100*262144/prefillTokPerSec)

	for _, r := range results {
		if !r.Helped() && r.Saved() < 0 {
			fmt.Printf("\n⚠️  %q cost %d tokens on this text. %s\n",
				r.Name, -r.Saved(), r.Why)
			fmt.Println("   Drop it for this corpus — a pass that breaks a BPE merge is worse " +
				"than no pass.")
		}
	}

	if *write != "" {
		if err := os.WriteFile(*write, []byte(vcontext.Canonicalise(string(raw))), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "canon:", err)
			return 1
		}
		fmt.Printf("\nwrote %s\n", *write)
	}
	return 0
}
