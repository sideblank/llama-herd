// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package engine

import (
	"context"
	"errors"
	"testing"
)

// scriptedDrafter proposes whatever it is told to, so acceptance and rejection can both be
// exercised deterministically.
type scriptedDrafter struct {
	propose  []Token // proposed every call
	max      int
	calls    int
	accepted []int
	released []SeqID
	err      error
}

func (d *scriptedDrafter) MaxDraft() int { return d.max }

func (d *scriptedDrafter) Draft(_ SeqID, _ Token, _ Pos, n int) ([]Token, error) {
	d.calls++
	if d.err != nil {
		return nil, d.err
	}
	out := d.propose
	if len(out) > n {
		out = out[:n]
	}
	return append([]Token(nil), out...), nil
}

func (d *scriptedDrafter) Accept(_ SeqID, accepted int, _ Token) error {
	d.accepted = append(d.accepted, accepted)
	return nil
}

func (d *scriptedDrafter) Release(seq SeqID) { d.released = append(d.released, seq) }

// A perfect drafter should produce several tokens from a single forward pass — which is the
// entire point, and shows up as tokens-per-pass above one.
func TestAcceptedDraftsProduceMultipleTokensPerPass(t *testing.T) {
	f := newFake(1, 16)
	// The target will emit a, b, c, d. The drafter proposes exactly that continuation,
	// so every proposal should be accepted.
	f.script[0] = []Token{'a', 'b', 'c', 'd', 'e', 'f'}
	d := &scriptedDrafter{propose: []Token{'b', 'c'}, max: 2}

	e := New(f, Config{Drafter: d})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{Prompt: "x", MaxTokens: 6})
	if err != nil {
		t.Fatal(err)
	}
	collect(t, s)

	st := e.Stats()
	if st.DecodePasses == 0 {
		t.Fatal("no passes recorded")
	}
	ratio := float64(st.TokensGenerated) / float64(st.DecodePasses)
	if ratio <= 1.0 {
		t.Fatalf("tokens per pass = %.2f with an accepting drafter; speculation produced nothing",
			ratio)
	}
	if d.calls == 0 {
		t.Fatal("drafter was never called")
	}
}

// A drafter whose proposals are wrong must not corrupt output: the target's own tokens stand,
// and the rejected tail is removed from the cache.
func TestRejectedDraftsAreRolledBackAndOutputIsUnchanged(t *testing.T) {
	f := newFake(1, 16)
	f.script[0] = []Token{'a', 'b', 'c'}
	// Proposals that never match what the target emits.
	d := &scriptedDrafter{propose: []Token{'z', 'z'}, max: 2}

	e := New(f, Config{Drafter: d})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{Prompt: "x", MaxTokens: 3})
	if err != nil {
		t.Fatal(err)
	}
	text, _ := collect(t, s)

	if text != "abc" {
		t.Fatalf("text = %q, want %q — a wrong draft must not change what is produced", text, "abc")
	}
	f.mu.Lock()
	trims := len(f.trims)
	f.mu.Unlock()
	if trims == 0 {
		t.Fatal("rejected drafts were never trimmed from the cache")
	}
	for _, n := range d.accepted {
		if n != 0 {
			t.Fatalf("accepted %d of a draft that never matched", n)
		}
	}
}

// Without a drafter the engine must behave exactly as before.
func TestNoDrafterMeansOneTokenPerPass(t *testing.T) {
	f := newFake(1, 16)
	f.script[0] = []Token{'a', 'b', 'c', 'd'}

	e := New(f, Config{})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{Prompt: "x", MaxTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	text, _ := collect(t, s)
	if text != "abcd" {
		t.Fatalf("text = %q", text)
	}
	st := e.Stats()
	if r := float64(st.TokensGenerated) / float64(st.DecodePasses); r > 1.01 {
		t.Fatalf("tokens per pass = %.2f without a drafter", r)
	}
}

// Speculation is an optimisation, so a failing drafter must degrade to ordinary decoding
// rather than failing the request.
func TestFailingDrafterDegradesInsteadOfFailing(t *testing.T) {
	f := newFake(1, 16)
	f.script[0] = []Token{'a', 'b'}
	d := &scriptedDrafter{max: 2, err: errors.New("draft model exploded")}

	e := New(f, Config{Drafter: d})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{Prompt: "x", MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	text, reason := collect(t, s)
	if text != "ab" {
		t.Fatalf("text = %q, want %q despite the drafter failing", text, "ab")
	}
	if reason == "" {
		t.Fatal("stream should still terminate normally")
	}
}

// Drafts consume batch budget like any other entry. If they did not, a speculating stream
// would overrun the batch, which the backend rejects outright.
func TestDraftsRespectTheBatchBudget(t *testing.T) {
	const cap = 4
	f := newFake(2, cap)
	for i := SeqID(0); i < 2; i++ {
		f.script[i] = make([]Token, 40)
		for j := range f.script[i] {
			f.script[i][j] = 'a'
		}
	}
	// Ask for more drafts than the batch could possibly hold.
	d := &scriptedDrafter{propose: []Token{'a', 'a', 'a', 'a', 'a', 'a'}, max: 6}

	e := New(f, Config{Drafter: d})
	defer run(t, e)()

	var streams []*Stream
	for i := 0; i < 2; i++ {
		s, err := e.Submit(context.Background(), Request{Prompt: "x", MaxTokens: 20})
		if err != nil {
			t.Fatal(err)
		}
		streams = append(streams, s)
	}
	for _, s := range streams {
		collect(t, s)
	}

	if f.maxSeen > cap {
		t.Fatalf("staged a batch of %d against a cap of %d — drafts must consume budget",
			f.maxSeen, cap)
	}
}

// A drafter holds per-sequence state, so a finished slot must release it or the next request
// on that slot inherits the previous one's context — the same class of bug as a sampler that
// is not reset.
func TestFinishedSlotReleasesDrafterState(t *testing.T) {
	f := newFake(1, 16)
	f.script[0] = []Token{'a'}
	d := &scriptedDrafter{propose: []Token{'a'}, max: 1}

	e := New(f, Config{Drafter: d})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{Prompt: "x", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	collect(t, s)

	if len(d.released) == 0 {
		t.Fatal("drafter state was never released for a finished sequence")
	}
	if d.released[0] != 0 {
		t.Fatalf("released sequence %d, want 0", d.released[0])
	}
}
