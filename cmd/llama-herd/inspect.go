// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/sideblank/llama-herd/internal/llama"
)

// inspect answers the question a model card cannot be trusted for: does this file actually
// carry what it claims? It loads metadata only, so it is fast and needs no GPU.
func inspect(args []string) int {
	fs := newFlagSet("inspect")
	all := fs.Bool("all", false, "print every metadata key")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: llama-herd inspect [--all] <model.gguf>")
		return 2
	}
	path := fs.Arg(0)

	llama.Backend()
	defer llama.BackendFree()

	// Vocab-only skips the weights entirely: this reads a 15 GB file's header without
	// paging in the tensors, so inspecting a model costs seconds rather than minutes.
	mp := llama.DefaultModelParams()
	mp.VocabOnly = true
	mp.NGPULayers = 0

	m, err := llama.LoadModel(path, mp)
	if err != nil {
		fmt.Fprintf(os.Stderr, "inspect: %v\n", err)
		return 1
	}
	defer m.Free()

	fmt.Printf("%s\n\n", path)
	fmt.Printf("  %s\n", m.Summary())

	sh := m.Shape()
	if sh.Valid() {
		fmt.Printf("  kv_heads=%d  head_dim=%d/%d", sh.HeadsKV, sh.KeyLength, sh.ValueLength)
		if sh.Hybrid() {
			fmt.Printf("  hybrid: %d of %d layers cache (every %d)",
				sh.KVLayers(), sh.Layers, sh.FullAttentionInterval)
		}
		fmt.Printf("\n  KV per token: %.0f KiB at f16, %.0f KiB at q8_0, %.0f KiB at q4_0\n",
			sh.KVBytesPerToken(llama.KVf16)/1024,
			sh.KVBytesPerToken(llama.KVq8)/1024,
			sh.KVBytesPerToken(llama.KVq4)/1024)
	} else {
		fmt.Println("  attention geometry incomplete — capacity cannot be computed")
	}

	if _, err := m.ChatTemplate(); err == nil {
		fmt.Println("  chat template: present")
	} else {
		fmt.Println("  chat template: ABSENT — chat requests for this model will be refused")
	}

	mtp := m.MTP()
	fmt.Println()
	if mtp.DeclaredLayers > 0 {
		fmt.Printf("  MTP: metadata declares %d layer(s) via %s\n", mtp.DeclaredLayers, mtp.MetaKey)
		fmt.Println("       Declared is not the same as present. Most redistributed quantizations")
		fmt.Println("       keep this metadata while dropping the tensors, so the only proof is a")
		fmt.Println("       real load with load_mtp enabled — watch the loader output for the")
		fmt.Println("       layers being read.")
	} else {
		fmt.Println("  MTP: none declared — speculative decoding through an MTP head is unavailable")
	}

	if *all {
		meta := m.MetaAll()
		keys := make([]string, 0, len(meta))
		for k := range meta {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		fmt.Printf("\n  metadata (%d keys):\n", len(keys))
		for _, k := range keys {
			v := meta[k]
			if len(v) > 100 {
				v = v[:100] + "..."
			}
			fmt.Printf("    %-44s %s\n", k, strings.ReplaceAll(v, "\n", " "))
		}
	}
	return 0
}
