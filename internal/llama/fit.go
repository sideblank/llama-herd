// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

import (
	"fmt"
	"strconv"
	"strings"
)

// KVPrecision is the per-element size of the KV cache.
type KVPrecision struct {
	Name  string
	Bytes float64
}

// The KV precisions worth considering. Lower precision is the largest single lever on how
// much context fits, and it costs quality that has to be measured rather than assumed.
var (
	KVf16 = KVPrecision{"f16", 2}
	KVq8  = KVPrecision{"q8_0", 1}
	KVq5  = KVPrecision{"q5_1", 0.75}
	KVq4  = KVPrecision{"q4_0", 0.5}
)

// Shape is what a model file declares about its attention geometry — the numbers that decide
// KV cost, which is what decides how many streams and how much context fit on a card.
type Shape struct {
	Arch        string
	Layers      int
	HeadsKV     int
	KeyLength   int
	ValueLength int
	CtxTrain    int
	EmbdLength  int

	// FullAttentionInterval is set by hybrid architectures, where only every Nth layer
	// keeps a KV cache and the rest use linear attention with a constant-size recurrent
	// state that does not grow with context.
	//
	// Ignoring it overstates KV cost by exactly this factor — enough to declare a working
	// configuration impossible. 0 or 1 means every layer caches.
	FullAttentionInterval int
}

// KVLayers is how many layers actually hold a KV cache.
func (s Shape) KVLayers() int {
	if s.FullAttentionInterval > 1 {
		n := s.Layers / s.FullAttentionInterval
		if s.Layers%s.FullAttentionInterval != 0 {
			n++
		}
		if n < 1 {
			n = 1
		}
		return n
	}
	return s.Layers
}

// Hybrid reports whether only some layers cache.
func (s Shape) Hybrid() bool { return s.FullAttentionInterval > 1 }

func (m *Model) metaInt(key string) int {
	v, ok := m.Meta(key)
	if !ok {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0
	}
	return n
}

// Shape reads the attention geometry from the model's metadata.
func (m *Model) Shape() Shape {
	a := m.Architecture()
	s := Shape{
		Arch:        a,
		Layers:      m.metaInt(a + ".block_count"),
		HeadsKV:     m.metaInt(a + ".attention.head_count_kv"),
		KeyLength:   m.metaInt(a + ".attention.key_length"),
		ValueLength: m.metaInt(a + ".attention.value_length"),
		CtxTrain:    m.metaInt(a + ".context_length"),
		EmbdLength:  m.metaInt(a + ".embedding_length"),

		FullAttentionInterval: m.metaInt(a + ".full_attention_interval"),
	}
	// Some files omit the explicit key/value lengths, in which case they are the
	// embedding size divided by the head count.
	if s.KeyLength == 0 && s.EmbdLength > 0 {
		if h := m.metaInt(a + ".attention.head_count"); h > 0 {
			s.KeyLength = s.EmbdLength / h
		}
	}
	if s.ValueLength == 0 {
		s.ValueLength = s.KeyLength
	}
	return s
}

// Valid reports whether enough was declared to compute a KV cost.
func (s Shape) Valid() bool {
	return s.Layers > 0 && s.HeadsKV > 0 && s.KeyLength > 0 && s.ValueLength > 0
}

// KVBytesPerToken is the cache cost of one token across all layers, at the given precision.
//
// This is the number the whole capacity question turns on: total context is VRAM left after
// the weights, divided by this. It is per token across every layer and both K and V, so it
// is much larger than intuition suggests.
func (s Shape) KVBytesPerToken(p KVPrecision) float64 {
	return float64(s.KVLayers()) * float64(s.HeadsKV) *
		(float64(s.KeyLength) + float64(s.ValueLength)) * p.Bytes
}

// MTPDraftLayers is how many layers the MTP draft context caches.
//
// Speculating from a model's own prediction head opens a second context over the same
// weights, and that context allocates its own KV cache — it does not share the target's. The
// weights are not duplicated, so the cost is KV alone, but it is charged at the same context
// length and precision as the target.
//
// This is the term that is easy to omit and expensive to omit: a plan that fits without it
// loads, reports healthy, and then dies on the first batch large enough to need compute
// buffers that are no longer there.
func (s Shape) MTPDraftLayers(declared int) int {
	if declared < 1 {
		return 0
	}
	// The head's layers cache like any other attention layer. A hybrid model's interval
	// applies to its body, not to the appended head, so these are counted in full.
	return declared
}

// MTPDraftBytes is the KV a draft context adds, for a given total context and precision.
func (s Shape) MTPDraftBytes(declared int, p KVPrecision, totalTokens int64) int64 {
	n := s.MTPDraftLayers(declared)
	if n == 0 || totalTokens <= 0 {
		return 0
	}
	perToken := float64(n) * float64(s.HeadsKV) *
		(float64(s.KeyLength) + float64(s.ValueLength)) * p.Bytes
	return int64(perToken * float64(totalTokens))
}

// FitInput describes a card and a model file.
type FitInput struct {
	Shape Shape
	// MTPLayers is the model's declared prediction-head layers. Non-zero only charges for a
	// draft context when Speculate is set, since loading a head and drafting from it are
	// different decisions with different costs.
	MTPLayers int
	// Speculate charges for the draft context an "mtp" speculation type would open.
	Speculate bool
	// WeightBytes is the size of the model file, which is what the weights occupy once
	// resident.
	WeightBytes uint64
	// VRAMBytes is the card's capacity.
	VRAMBytes uint64
	// OverheadBytes is compute buffers, the CUDA context and fragmentation. Ignoring it
	// produces a plan that fits on paper and fails on the card.
	OverheadBytes uint64
}

// DefaultOverhead is a deliberately conservative reservation for compute buffers, the CUDA
// context, and allocator fragmentation.
const DefaultOverhead = 1536 << 20

// FitResult is the capacity available for context.
type FitResult struct {
	Precision   KVPrecision
	KVBudget    int64 // bytes available for the KV cache
	PerToken    float64
	TotalTokens int64 // total context across every stream
	// MTPPerToken is the share of PerToken belonging to an MTP draft context, so a caller
	// can report what speculation costs rather than only that it was charged.
	MTPPerToken float64
}

// Fit computes how much total context fits, at one KV precision.
func Fit(in FitInput, p KVPrecision) FitResult {
	overhead := in.OverheadBytes
	if overhead == 0 {
		overhead = DefaultOverhead
	}
	budget := int64(in.VRAMBytes) - int64(in.WeightBytes) - int64(overhead)
	r := FitResult{Precision: p, KVBudget: budget, PerToken: in.Shape.KVBytesPerToken(p)}
	// A draft context caches the head's layers over the same context, so the per-token cost
	// rises rather than the budget falling. Charging it here keeps every derived figure —
	// total tokens, the per-stream plan, the weight budget — consistent with it.
	if in.Speculate && in.MTPLayers > 0 {
		r.MTPPerToken = float64(in.Shape.MTPDraftLayers(in.MTPLayers)) *
			float64(in.Shape.HeadsKV) *
			(float64(in.Shape.KeyLength) + float64(in.Shape.ValueLength)) * p.Bytes
		r.PerToken += r.MTPPerToken
	}
	if budget > 0 && r.PerToken > 0 {
		r.TotalTokens = int64(float64(budget) / r.PerToken)
	}
	return r
}

// Plan is one streams-by-context arrangement.
type Plan struct {
	Streams   int
	PerStream int64
	Total     int64
	Fits      bool
	ShortBy   int64 // bytes short when it does not fit
}

// PlanFor reports whether a given number of streams at a given context each will fit, and by
// how much it misses when it does not. Reporting the shortfall matters: "does not fit" is far
// less useful than "needs another 9 GB", which tells you whether a second card closes it.
func PlanFor(in FitInput, p KVPrecision, streams int, perStream int64) Plan {
	total := int64(streams) * perStream
	// Take the per-token cost from Fit rather than from the shape, so a charged draft
	// context is included here too. Reading it from the shape leaves this section quietly
	// disagreeing with the table above it — the same target reported as fitting by one and
	// short by the other, with nothing to say which is right.
	f := Fit(in, p)
	need := int64(float64(total) * f.PerToken)
	have := f.KVBudget

	pl := Plan{Streams: streams, PerStream: perStream, Total: total}
	if have >= need {
		pl.Fits = true
	} else {
		pl.ShortBy = need - have
	}
	return pl
}

// WeightBudget is what a target leaves for the weights, and what that implies about how far
// they must be compressed.
type WeightBudget struct {
	// Bytes available for weights after KV and overhead.
	Bytes int64
	// BitsPerWeight the model must reach to fit, if the parameter count is known. This is
	// the number that decides whether a target is reachable: below roughly 3 bits quality
	// degrades sharply on most models, and below 2 it is rarely usable.
	BitsPerWeight float64
	// Params is the parameter count used, or 0 if it could not be determined.
	Params int64
}

// BudgetFor reports what remains for weights once a streams-by-context target is met.
//
// Quantization compresses weights, not the KV cache, so this is the only budget it can move.
// A target whose KV cost already exceeds the card is unreachable at any quantization.
func BudgetFor(in FitInput, p KVPrecision, streams int, perStream int64, params int64) WeightBudget {
	// Includes the draft context when one is charged, for the same reason as PlanFor.
	kvNeed := int64(float64(int64(streams)*perStream) * Fit(in, p).PerToken)
	overhead := in.OverheadBytes
	if overhead == 0 {
		overhead = DefaultOverhead
	}
	b := WeightBudget{Bytes: int64(in.VRAMBytes) - kvNeed - int64(overhead), Params: params}
	if params > 0 && b.Bytes > 0 {
		b.BitsPerWeight = float64(b.Bytes) * 8 / float64(params)
	}
	return b
}

// Verdict describes whether a required bits-per-weight is realistic.
//
// These thresholds are guidance for planning, not measurements. Whether a specific model
// survives a given level has to be measured on that model — some tolerate 3 bits well and
// others fall apart, and the difference does not follow from parameter count.
func (b WeightBudget) Verdict() string {
	switch {
	case b.Bytes <= 0:
		return "unreachable — the KV cache alone exceeds the card"
	case b.Params == 0:
		return "parameter count unknown; compare the byte budget against candidate quantizations"
	case b.BitsPerWeight >= 8:
		return "comfortable — 8-bit or better"
	case b.BitsPerWeight >= 5:
		return "comfortable — around Q5/Q6"
	case b.BitsPerWeight >= 4:
		return "workable — around Q4, the usual target"
	case b.BitsPerWeight >= 3:
		return "tight — Q3 territory, quality must be measured rather than assumed"
	case b.BitsPerWeight >= 2:
		return "severe — Q2 territory, often unusable; verify before committing GPU time"
	default:
		return "not reachable at any usable quantization"
	}
}

// ParamCount reads the declared parameter count, or 0 when the file does not state one.
//
// Files state this two ways: an exact count, or a human label like "0.5B" or "35B". Both are
// accepted, since most published quantizations carry only the label.
func (m *Model) ParamCount() int64 {
	if v, ok := m.Meta("general.parameter_count"); ok {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			return n
		}
	}
	if v, ok := m.Meta("general.size_label"); ok {
		if n := parseSizeLabel(v); n > 0 {
			return n
		}
	}
	return 0
}

// parseSizeLabel turns "0.5B", "35B", "8x7B" or "700M" into a parameter count.
func parseSizeLabel(s string) int64 {
	s = strings.TrimSpace(strings.ToUpper(s))
	// Mixture labels like "8x7B" describe total parameters as the product.
	mult := 1.0
	if i := strings.IndexByte(s, 'X'); i > 0 {
		if n, err := strconv.ParseFloat(s[:i], 64); err == nil && n > 0 {
			mult, s = n, s[i+1:]
		}
	}
	var scale float64
	switch {
	case strings.HasSuffix(s, "B"):
		scale, s = 1e9, strings.TrimSuffix(s, "B")
	case strings.HasSuffix(s, "M"):
		scale, s = 1e6, strings.TrimSuffix(s, "M")
	default:
		return 0
	}
	n, err := strconv.ParseFloat(s, 64)
	if err != nil || n <= 0 {
		return 0
	}
	return int64(mult * n * scale)
}

// GiB renders a byte count for humans.
func GiB(b int64) string {
	if b < 0 {
		return fmt.Sprintf("-%.1f GiB", float64(-b)/(1<<30))
	}
	return fmt.Sprintf("%.1f GiB", float64(b)/(1<<30))
}
