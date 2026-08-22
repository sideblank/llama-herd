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

// The acceptance rate must be counted, not inferred from tokens per pass — that ratio moves
// with prompt length and stream mix, and reads below one on a long prompt with a short answer.
func TestAcceptanceRateIsCountedDirectly(t *testing.T) {
	f := newFake(1, 32)
	f.script[0] = []Token{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}
	// Proposes 'b' then 'z'. The first matches what the target emits, the second does not,
	// so exactly half of each round's proposals should be accepted.
	d := &scriptedDrafter{propose: []Token{'b', 'z'}, max: 2}

	e := New(f, Config{Drafter: d})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{Prompt: "x", MaxTokens: 8})
	if err != nil {
		t.Fatal(err)
	}
	collect(t, s)

	st := e.Stats()
	if st.DraftsProposed == 0 {
		t.Fatal("no drafts were counted as proposed")
	}
	if st.DraftsAccepted == 0 {
		t.Fatal("no drafts were counted as accepted, though the first proposal matches")
	}
	if st.DraftsAccepted > st.DraftsProposed {
		t.Fatalf("accepted %d of %d proposed", st.DraftsAccepted, st.DraftsProposed)
	}
	if r := st.AcceptanceRate(); r <= 0 || r > 1 {
		t.Fatalf("acceptance rate = %.2f, want a fraction", r)
	}
}

func TestNoDrafterProposesNothing(t *testing.T) {
	f := newFake(1, 16)
	f.script[0] = []Token{'a', 'b'}
	e := New(f, Config{})
	defer run(t, e)()

	s, _ := e.Submit(context.Background(), Request{Prompt: "x", MaxTokens: 2})
	collect(t, s)

	st := e.Stats()
	if st.DraftsProposed != 0 || st.DraftsAccepted != 0 {
		t.Fatalf("proposed=%d accepted=%d without a drafter", st.DraftsProposed, st.DraftsAccepted)
	}
	if st.AcceptanceRate() != 0 {
		t.Fatal("acceptance rate should be zero when nothing was proposed")
	}
}

// observingDrafter records how often it was shown a decode, to prove a state-predicting
// drafter sees every pass rather than only the ones where a draft is wanted.
type observingDrafter struct {
	scriptedDrafter
	observed int
}

func (d *observingDrafter) ObserveDecode() error { d.observed++; return nil }

// A drafter predicting from the target's internal state must be shown every decode. Showing
// it only when drafting would leave it working from state several steps stale, and its
// proposals would quietly stop being accepted with nothing to explain why.
func TestStatePredictingDrafterSeesEveryDecode(t *testing.T) {
	f := newFake(1, 16)
	f.script[0] = []Token{'a', 'b', 'c', 'd'}
	d := &observingDrafter{scriptedDrafter: scriptedDrafter{propose: []Token{'x'}, max: 1}}

	e := New(f, Config{Drafter: d})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{Prompt: "hello", MaxTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	collect(t, s)

	st := e.Stats()
	if d.observed == 0 {
		t.Fatal("the drafter was never shown a decode")
	}
	// Every pass must be observed, including the prefill passes where no draft is wanted.
	if uint64(d.observed) != st.DecodePasses {
		t.Fatalf("observed %d decodes but %d passes ran — a state-predicting drafter must see "+
			"all of them", d.observed, st.DecodePasses)
	}
}

// posRecordingDrafter records the position it was told for each draft.
type posRecordingDrafter struct {
	positions []Pos
	lasts     []Token
}

func (d *posRecordingDrafter) Draft(_ SeqID, last Token, pos Pos, _ int) ([]Token, error) {
	d.positions = append(d.positions, pos)
	d.lasts = append(d.lasts, last)
	return nil, nil
}
func (d *posRecordingDrafter) Accept(SeqID, int, Token) error { return nil }
func (d *posRecordingDrafter) Release(SeqID)                  {}
func (d *posRecordingDrafter) MaxDraft() int                  { return 4 }

// A drafter is told the position OF the token it is continuing from, not the position after
// it. A head that decodes into its own KV cache lines up against this, and one position too
// high puts that cache permanently ahead of where the target asks it to draft — which the
// backend reports as inconsistent sequence positions, not as a bad draft.
func TestDrafterIsToldThePositionOfTheLastToken(t *testing.T) {
	f := newFake(1, 16)
	f.script[0] = []Token{'a', 'b', 'c'}
	d := &posRecordingDrafter{}

	e := New(f, Config{Drafter: d})
	defer run(t, e)()

	const prompt = "hello"
	s, err := e.Submit(context.Background(), Request{Prompt: prompt, MaxTokens: 3})
	if err != nil {
		t.Fatal(err)
	}
	collect(t, s)

	if len(d.positions) == 0 {
		t.Fatal("the drafter was never asked for a draft")
	}
	// The first generated token sits immediately after the prompt, so that is the position
	// the drafter must be given the first time it is asked.
	want := Pos(len([]byte(prompt)))
	if got := d.positions[0]; got != want {
		t.Fatalf("first draft position = %d, want %d (the position of the token being "+
			"continued, not the one after it)", got, want)
	}
	for i := 1; i < len(d.positions); i++ {
		if d.positions[i] <= d.positions[i-1] {
			t.Fatalf("draft position went backwards or stalled: %v", d.positions)
		}
	}
}
