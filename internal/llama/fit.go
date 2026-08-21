// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

import (
	"fmt"
	"strconv"
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
}

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
	return float64(s.Layers) * float64(s.HeadsKV) *
		(float64(s.KeyLength) + float64(s.ValueLength)) * p.Bytes
}

// FitInput describes a card and a model file.
type FitInput struct {
	Shape Shape
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
}

// Fit computes how much total context fits, at one KV precision.
func Fit(in FitInput, p KVPrecision) FitResult {
	overhead := in.OverheadBytes
	if overhead == 0 {
		overhead = DefaultOverhead
	}
	budget := int64(in.VRAMBytes) - int64(in.WeightBytes) - int64(overhead)
	r := FitResult{Precision: p, KVBudget: budget, PerToken: in.Shape.KVBytesPerToken(p)}
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
	need := int64(float64(total) * in.Shape.KVBytesPerToken(p))
	have := Fit(in, p).KVBudget

	pl := Plan{Streams: streams, PerStream: perStream, Total: total}
	if have >= need {
		pl.Fits = true
	} else {
		pl.ShortBy = need - have
	}
	return pl
}

// GiB renders a byte count for humans.
func GiB(b int64) string {
	if b < 0 {
		return fmt.Sprintf("-%.1f GiB", float64(-b)/(1<<30))
	}
	return fmt.Sprintf("%.1f GiB", float64(b)/(1<<30))
}
