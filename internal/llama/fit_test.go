package llama

import "testing"

// qwenLike is a hybrid 35B-A3B-shaped model: only every fourth layer caches, which is the
// property that once made a hand calculation declare a running configuration impossible.
func qwenLike() Shape {
	return Shape{
		Arch: "qwen3moe", Layers: 40, HeadsKV: 4,
		KeyLength: 128, ValueLength: 128, CtxTrain: 262144,
		FullAttentionInterval: 4,
	}
}

func TestHybridCachesEveryNthLayer(t *testing.T) {
	s := qwenLike()
	if got := s.KVLayers(); got != 10 {
		t.Fatalf("KVLayers = %d, want 10 of %d", got, s.Layers)
	}
	dense := s
	dense.FullAttentionInterval = 0
	if s.KVBytesPerToken(KVq8)*4 != dense.KVBytesPerToken(KVq8) {
		t.Fatal("hybrid KV cost should be exactly a quarter of the dense equivalent")
	}
}

// The draft context an "mtp" speculation type opens caches its own layers over the same
// context. A plan that omits it fits on paper and dies on the first large batch, so the
// charge must actually change the answer.
func TestMTPDraftContextIsCharged(t *testing.T) {
	base := FitInput{
		Shape:       qwenLike(),
		WeightBytes: 16 << 30,
		VRAMBytes:   24 << 30,
		MTPLayers:   1,
	}
	without := Fit(base, KVq8)

	spec := base
	spec.Speculate = true
	with := Fit(spec, KVq8)

	if with.MTPPerToken <= 0 {
		t.Fatal("speculating should charge a per-token cost for the draft context")
	}
	if with.TotalTokens >= without.TotalTokens {
		t.Fatalf("charging a draft context should reduce capacity: %d -> %d",
			without.TotalTokens, with.TotalTokens)
	}
	// One head layer against ten caching body layers: an eleventh of the total, so the
	// charge is real but modest. A result far from this means the head is being counted
	// against the wrong layer base.
	ratio := with.MTPPerToken / without.PerToken
	if ratio < 0.05 || ratio > 0.2 {
		t.Fatalf("draft context is %.3f of target KV; expected about a tenth", ratio)
	}
}

// Declaring a head is not the same decision as drafting from it: load_mtp makes it resident,
// speculation opens the context. Charging on declaration alone would refuse configurations
// that work.
func TestDeclaredHeadIsNotChargedUnlessSpeculating(t *testing.T) {
	in := FitInput{
		Shape: qwenLike(), WeightBytes: 16 << 30, VRAMBytes: 24 << 30, MTPLayers: 1,
	}
	if got := Fit(in, KVq8).MTPPerToken; got != 0 {
		t.Fatalf("MTPPerToken = %v with Speculate false, want 0", got)
	}
}

func TestNoHeadMeansNoCharge(t *testing.T) {
	in := FitInput{
		Shape: qwenLike(), WeightBytes: 16 << 30, VRAMBytes: 24 << 30,
		MTPLayers: 0, Speculate: true,
	}
	if got := Fit(in, KVq8).MTPPerToken; got != 0 {
		t.Fatalf("MTPPerToken = %v with no declared layers, want 0", got)
	}
}
