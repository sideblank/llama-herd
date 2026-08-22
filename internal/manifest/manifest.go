// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package manifest describes which models to host and how to size each one.
//
// Sizing is per model rather than global on purpose: a herd may span cards of different
// capacities, and one stream count applied to all of them either wastes the large card or
// fails on the small one.
package manifest

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

// Manifest is the top-level document.
type Manifest struct {
	// Listen is the address the server binds. Defaults to :8080.
	Listen string  `json:"listen,omitempty"`
	Models []Model `json:"models"`
}

// Model is one entry of the herd.
type Model struct {
	// Name is how requests address this model. It need not match the filename.
	Name string `json:"name"`
	// Path is the GGUF file.
	Path string `json:"path"`

	// GPULayers is how many layers to offload. -1 means all, 0 means CPU only.
	GPULayers int32 `json:"gpu_layers"`
	// MainGPU is the device used when SplitMode is empty or "none".
	MainGPU int32 `json:"main_gpu,omitempty"`
	// SplitMode is one of none, layer, row, tensor.
	SplitMode string `json:"split_mode,omitempty"`
	// TensorSplit is the share each device receives. Give one value per device.
	// Leave empty for an even split — which is wrong on a mixed host.
	TensorSplit []float32 `json:"tensor_split,omitempty"`

	// Context is the total context across all streams for this model.
	Context uint32 `json:"context"`

	// AdmitContext caps what one request may occupy, below the context allocated per
	// stream. Zero admits the whole window.
	//
	// Cache memory is reserved up front from `context` whether or not requests can fill it,
	// so admitting less costs nothing and buys certainty: a stream that cannot reach the end
	// of its window cannot be evicted from it. The refusal happens at submit time, with a
	// number, instead of mid-answer.
	AdmitContext uint32 `json:"admit_context,omitempty"`
	// Batch is the largest number of tokens one decode pass may carry, and therefore
	// the scheduler's per-tick budget.
	Batch uint32 `json:"batch,omitempty"`
	// Streams is how many concurrent generations this model can serve.
	Streams uint32 `json:"streams"`

	Threads      int32 `json:"threads,omitempty"`
	ThreadsBatch int32 `json:"threads_batch,omitempty"`

	// LoadMTP loads multi-token-prediction layers when the file carries them.
	LoadMTP bool `json:"load_mtp,omitempty"`

	// KVTypeK and KVTypeV set the KV cache precision, separately for keys and values.
	// One of f16, q8_0, q5_1, q4_0.
	//
	// Keys tolerate quantization less well than values, so K at q8 with V at q4 is often
	// a better trade than putting both at the same level — it saves a quarter of the
	// cache for less quality cost. That combination needs flash attention AND a build
	// with all flash-attention quant kernels compiled; without them it falls off the fast
	// path silently.
	KVTypeK string `json:"kv_type_k,omitempty"`
	KVTypeV string `json:"kv_type_v,omitempty"`

	// FlashAttention is required by any quantized KV type. Leaving it off with a
	// quantized cache does not work.
	FlashAttention bool `json:"flash_attention,omitempty"`

	// KVUnified shares one KV pool across streams instead of splitting it evenly, so a
	// single request may exceed its even share and use whatever is free.
	//
	// EXPERIMENTAL, OFF BY DEFAULT, AND NOT TESTED UNDER LOAD.
	//
	// It works, in that the setting reaches the context and the per-stream ceiling becomes
	// the whole pool. What has not been done is the scheduling change it needs: admission
	// still checks a per-stream ceiling, which is a real reservation under a split but is
	// the entire cache under one pool. Several requests can therefore each be admitted
	// believing they may use everything, and the herd evicts its way out of the
	// overcommitment — visible to a user as answers truncating for no stated reason.
	//
	// Enable it if you want one long request to use idle capacity and you can tolerate
	// that. Leave it off for a fixed four-by-128k herd, which is the tested arrangement.
	KVUnified bool `json:"kv_unified,omitempty"`

	// MMProjPath is the multimodal projector accompanying a vision model. Without it the
	// model is text-only even if its weights support images.
	MMProjPath string `json:"mmproj_path,omitempty"`
	// VisionGPU offloads the vision encoder. Encoding an image on CPU is slow enough to
	// dominate time-to-first-token.
	VisionGPU bool `json:"vision_gpu,omitempty"`

	// MaxQueue bounds requests waiting for a stream. 0 is unbounded.
	MaxQueue int `json:"max_queue,omitempty"`

	Sampling Sampling `json:"sampling,omitempty"`

	// Speculation configures drafting for this model. Omitted means none.
	Speculation *Speculation `json:"speculation,omitempty"`
}

// Speculation selects a draft source and how far ahead it proposes.
type Speculation struct {
	// Type is the draft source.
	//
	//   lookup — predicts from the sequence's own context. Costs no memory and works on
	//            any model, but contributes only where output repeats context.
	//   mtp    — uses the model's own trained prediction head. Costs the memory that head
	//            occupies and works on any output, but only for models that carry one and
	//            quantizations that kept it.
	//
	// A model whose head was stripped in quantization falls back rather than failing, so
	// check the acceptance rate rather than assuming it is running.
	Type string `json:"type"`
	// MaxDraft bounds tokens proposed per step. Larger drafts win more when they land and
	// waste more batch space when they do not, so this trades against stream count.
	MaxDraft int `json:"max_draft,omitempty"`
	// Pattern is the match length for lookup drafting. Shorter patterns fire constantly
	// and are usually wrong; longer ones rarely fire outside heavily repeated text.
	Pattern int `json:"pattern,omitempty"`
}

// Sampling mirrors the sampler settings, with pointers so an omitted field is
// distinguishable from a deliberate zero — a temperature of 0 means greedy, not "default".
type Sampling struct {
	Temperature   *float32 `json:"temperature,omitempty"`
	TopK          *int32   `json:"top_k,omitempty"`
	TopP          *float32 `json:"top_p,omitempty"`
	MinP          *float32 `json:"min_p,omitempty"`
	RepeatLastN   *int32   `json:"repeat_last_n,omitempty"`
	RepeatPenalty *float32 `json:"repeat_penalty,omitempty"`
	Seed          *uint32  `json:"seed,omitempty"`
}

// Valid split modes.
const (
	SplitNone   = "none"
	SplitLayer  = "layer"
	SplitRow    = "row"
	SplitTensor = "tensor"
)

// Load reads and validates a manifest file.
func Load(path string) (*Manifest, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("manifest: %w", err)
	}
	defer f.Close()
	return Parse(f)
}

// Parse reads and validates a manifest.
func Parse(r io.Reader) (*Manifest, error) {
	var m Manifest
	dec := json.NewDecoder(r)
	// Strict here, unlike the HTTP surface: a misspelled key in an operator-authored
	// config should fail loudly rather than silently leaving a model at its default.
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return nil, fmt.Errorf("manifest: %w", err)
	}
	if err := m.Validate(); err != nil {
		return nil, err
	}
	m.applyDefaults()
	return &m, nil
}

func (m *Manifest) applyDefaults() {
	if m.Listen == "" {
		m.Listen = ":8080"
	}
	for i := range m.Models {
		mm := &m.Models[i]
		if mm.Streams == 0 {
			mm.Streams = 1
		}
		if mm.Batch == 0 {
			// A batch smaller than the context still works — prefill is chunked —
			// but it must be able to hold one token per stream, or a full herd
			// cannot even be scheduled in one pass.
			mm.Batch = 2048
		}
		if mm.SplitMode == "" {
			mm.SplitMode = SplitNone
		}
	}
}

// Validate reports every problem it finds rather than only the first, so a broken manifest
// takes one edit to fix instead of several rounds.
func (m *Manifest) Validate() error {
	var problems []string

	if len(m.Models) == 0 {
		problems = append(problems, "no models declared")
	}

	seen := map[string]bool{}
	for i, mm := range m.Models {
		where := fmt.Sprintf("models[%d]", i)
		if mm.Name != "" {
			where = fmt.Sprintf("models[%d] (%s)", i, mm.Name)
		}

		if mm.Name == "" {
			problems = append(problems, where+": name is required")
		} else if seen[mm.Name] {
			problems = append(problems, where+": duplicate name — requests address models by name")
		}
		seen[mm.Name] = true

		if mm.Path == "" {
			problems = append(problems, where+": path is required")
		}
		if mm.Context == 0 {
			problems = append(problems, where+": context is required")
		}

		for label, v := range map[string]string{"kv_type_k": mm.KVTypeK, "kv_type_v": mm.KVTypeV} {
			switch v {
			case "", "f16", "q8_0", "q5_1", "q4_0":
			default:
				problems = append(problems, fmt.Sprintf(
					"%s: %s %q is not one of f16, q8_0, q5_1, q4_0", where, label, v))
			}
		}
		if (mm.KVTypeK != "" && mm.KVTypeK != "f16") || (mm.KVTypeV != "" && mm.KVTypeV != "f16") {
			if !mm.FlashAttention {
				problems = append(problems, where+
					": quantized KV requires flash_attention — without it the setting does not work")
			}
		}

		switch mm.SplitMode {
		case "", SplitNone, SplitLayer, SplitRow, SplitTensor:
		default:
			problems = append(problems, fmt.Sprintf(
				"%s: split_mode %q is not one of none, layer, row, tensor", where, mm.SplitMode))
		}

		if len(mm.TensorSplit) > 0 {
			var sum float32
			for _, v := range mm.TensorSplit {
				if v < 0 {
					problems = append(problems, where+": tensor_split values must not be negative")
					break
				}
				sum += v
			}
			if sum == 0 {
				problems = append(problems, where+": tensor_split sums to zero, which offloads nothing")
			}
			// Validation runs before defaults, so an omitted split_mode is still
			// the empty string here — and omitting it is the common case.
			if mm.SplitMode == "" || mm.SplitMode == SplitNone {
				problems = append(problems, where+
					": tensor_split has no effect unless split_mode is layer, row or tensor")
			}
		}

		if sp := mm.Speculation; sp != nil {
			switch sp.Type {
			case "", "none", "lookup", "mtp":
			default:
				problems = append(problems, fmt.Sprintf(
					"%s: speculation type %q is not one of none, lookup, mtp", where, sp.Type))
			}
			if sp.Type == "mtp" && !mm.LoadMTP {
				problems = append(problems, where+
					": speculation type mtp needs load_mtp true, or the head is not resident to drive")
			}
			// Every draft occupies a batch entry, so a full herd drafting at once needs
			// room for all of it or the batch overruns.
			if sp.MaxDraft > 0 && mm.Streams > 0 && mm.Batch > 0 {
				need := mm.Streams * uint32(sp.MaxDraft+1)
				if need > mm.Batch {
					problems = append(problems, fmt.Sprintf(
						"%s: %d streams drafting %d tokens each needs a batch of %d, but batch is %d",
						where, mm.Streams, sp.MaxDraft, need, mm.Batch))
				}
			}
		}

		if mm.Streams > 0 && mm.Batch > 0 && uint32(mm.Batch) < mm.Streams {
			problems = append(problems, fmt.Sprintf(
				"%s: batch %d is smaller than streams %d — a full herd cannot decode in one pass",
				where, mm.Batch, mm.Streams))
		}

		if mm.Streams > 0 && mm.Context > 0 && mm.Context/mm.Streams < 256 {
			problems = append(problems, fmt.Sprintf(
				"%s: context %d across %d streams leaves %d tokens each, too little to be useful",
				where, mm.Context, mm.Streams, mm.Context/mm.Streams))
		}

		// Admitting more than a stream owns is the mistake this field exists to prevent: it
		// reads as a larger limit while the cache still holds what it always did, so the
		// request is accepted and then evicted part way through its answer.
		if mm.AdmitContext > 0 && mm.Streams > 0 && mm.Context > 0 {
			perStream := mm.Context / mm.Streams
			if mm.AdmitContext > perStream {
				problems = append(problems, fmt.Sprintf(
					"%s: admit_context %d exceeds the %d tokens each of %d streams owns — "+
						"admitting more than is allocated evicts mid-answer instead of refusing at submit",
					where, mm.AdmitContext, perStream, mm.Streams))
			}
		}
	}

	if len(problems) > 0 {
		return errors.New("manifest:\n  - " + strings.Join(problems, "\n  - "))
	}
	return nil
}
