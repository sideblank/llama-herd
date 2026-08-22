// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package engine

import (
	"context"
	"errors"
	"fmt"
	"sync"
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

// rewindFake wraps the fake backend with a cache that cannot be rewound by position, which
// is what recurrent and hybrid architectures present. It models the part that matters: the
// snapshot is taken before a batch is decoded, so restoring it discards the accepted tokens
// too, and they have to be replayed.
type rewindFake struct {
	*fakeBackend
	// applied is the token sequence the model has actually been walked through. It is what
	// a real recurrent state would encode, and it is what must match a non-speculative run.
	applied   []Token
	pending   []Token
	snapshot  []Token
	haveSnap  bool
	rollbacks int
}

// Staging is not applying. A checkpoint taken while a batch is being built captures the
// state before that batch has been through the model, so the distinction is the whole point
// of this stand-in: recording at stage time would make the snapshot look like it already
// contained the token it is meant to precede.
func (r *rewindFake) BatchAdd(tok Token, pos Pos, seq SeqID, wantLogits bool) error {
	if err := r.fakeBackend.BatchAdd(tok, pos, seq, wantLogits); err != nil {
		return err
	}
	r.pending = append(r.pending, tok)
	return nil
}

func (r *rewindFake) BatchClear() {
	r.pending = r.pending[:0]
	r.fakeBackend.BatchClear()
}

func (r *rewindFake) Decode() error {
	if err := r.fakeBackend.Decode(); err != nil {
		return err
	}
	r.applied = append(r.applied, r.pending...)
	r.pending = r.pending[:0]
	return nil
}

func (r *rewindFake) TrimSeq(SeqID, Pos) bool { return false }

func (r *rewindFake) Checkpoint(SeqID) error {
	r.snapshot = append(r.snapshot[:0], r.applied...)
	r.haveSnap = true
	return nil
}

func (r *rewindFake) Rollback(_ SeqID, _ Pos) error {
	if !r.haveSnap {
		return fmt.Errorf("no checkpoint")
	}
	r.applied = append(r.applied[:0], r.snapshot...)
	r.rollbacks++
	return nil
}

func (r *rewindFake) DropCheckpoint(SeqID) { r.haveSnap = false }

// Speculation is an optimisation and must not change what the model is asked to produce. On
// an architecture that cannot rewind by position, a rollback restores state from before the
// accepted tokens as well as the rejected ones — so the accepted ones have to be walked
// through again. Skipping that leaves the caches disagreeing, and the only symptom is output
// that quietly degrades: nothing errors, and the text still reads like text.
func TestRollbackReplaysAcceptedTokens(t *testing.T) {
	script := []Token{'a', 'b', 'c', 'd', 'e'}

	// A drafter that is always wrong, so every step rejects and rolls back.
	plain := &rewindFake{fakeBackend: newFake(1, 64)}
	plain.script[0] = script
	base := New(plain, Config{})
	defer run(t, base)()
	bs, err := base.Submit(context.Background(), Request{Prompt: "hi", MaxTokens: 5})
	if err != nil {
		t.Fatal(err)
	}
	wantText, _ := collect(t, bs)

	const prompt = "hi"
	spec := &rewindFake{fakeBackend: newFake(1, 64)}
	spec.script[0] = script
	d := &scriptedDrafter{propose: []Token{'z', 'z'}, max: 2}
	e := New(spec, Config{Drafter: d, NeedsRewind: true})
	defer run(t, e)()
	ss, err := e.Submit(context.Background(), Request{Prompt: prompt, MaxTokens: 5})
	if err != nil {
		t.Fatal(err)
	}
	gotText, _ := collect(t, ss)

	if gotText != wantText {
		t.Fatalf("speculation changed the output: got %q, want %q", gotText, wantText)
	}
	if spec.rollbacks == 0 {
		t.Fatal("no rollback happened, so this proves nothing")
	}

	// The real assertion. Speculation must walk the model through exactly the same tokens
	// as a run without it. Text alone cannot show this: state and output are separate, and
	// it is precisely when they diverge that the output stays readable while being wrong.
	want := plain.applied
	// The speculative run may be one step behind: the last step rolls back and the request
	// ends before replaying it, which is harmless because the slot is released. Anything
	// more than that means state was lost or corrupted.
	if len(spec.applied) < len(want)-1 {
		t.Fatalf("speculation walked the model through %d tokens, a plain run used %d — "+
			"accepted tokens were rolled back and never replayed\n spec:  %v\n plain: %v",
			len(spec.applied), len(want), spec.applied, want)
	}
	if len(spec.applied) > len(want) {
		t.Fatalf("speculation applied more tokens than a plain run — something was replayed "+
			"twice\n spec:  %v\n plain: %v", spec.applied, want)
	}
	for i := range spec.applied {
		if spec.applied[i] != want[i] {
			t.Fatalf("model state diverged at %d\n spec:  %v\n plain: %v", i, spec.applied, want)
		}
	}
}

// Checkpoints are per-sequence state, like samplers and drafters before them. One sequence
// rolling back must not disturb another's, and a single-stream test cannot show that: the
// failure needs two slots speculating at once, which is the arrangement this engine is for.
func TestCheckpointsAreIsolatedBetweenSequences(t *testing.T) {
	f := newFake(2, 64)
	f.script[0] = []Token{'a', 'b', 'c'}
	f.script[1] = []Token{'x', 'y', 'z'}

	rw := &recordingRewinder{}
	e := New(f, Config{Drafter: &scriptedDrafter{propose: []Token{'!'}, max: 1}, NeedsRewind: true})
	e.rewinder = rw
	defer run(t, e)()

	var wg sync.WaitGroup
	out := make([]string, 2)
	for i := range out {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			s, err := e.Submit(context.Background(), Request{Prompt: "hi", MaxTokens: 3})
			if err != nil {
				t.Error(err)
				return
			}
			out[i], _ = collect(t, s)
		}(i)
	}
	wg.Wait()

	rw.mu.Lock()
	defer rw.mu.Unlock()
	if len(rw.ops) == 0 {
		t.Fatal("no checkpoint activity, so this proves nothing")
	}
	// Replay the operations in order rather than inspecting the end state: a checkpoint is
	// live only between the call that takes it and the drop that releases it, so the final
	// state says nothing about whether each rollback had one at the time.
	live := map[SeqID]bool{}
	for _, op := range rw.ops {
		switch op.kind {
		case "checkpoint":
			live[op.seq] = true
		case "rollback":
			if !live[op.seq] {
				t.Fatalf("sequence %d rolled back with no live checkpoint: %v", op.seq, rw.ops)
			}
		case "drop":
			// A slot's checkpoint must be released when it finishes, or a reused slot
			// could roll back into a finished request's state.
			delete(live, op.seq)
		}
	}
	for seq := range live {
		t.Fatalf("sequence %d kept its checkpoint past the end of its request: %v", seq, rw.ops)
	}
	// Both sequences must have participated, or the isolation was never exercised.
	seen := map[SeqID]bool{}
	for _, op := range rw.ops {
		seen[op.seq] = true
	}
	if len(seen) < 2 {
		t.Fatalf("only %d sequence(s) checkpointed, so isolation was not tested: %v",
			len(seen), rw.ops)
	}
}

type rewOp struct {
	kind string
	seq  SeqID
}

type recordingRewinder struct {
	mu           sync.Mutex
	ops          []rewOp
	checkpointed map[SeqID]bool
}

func (r *recordingRewinder) Checkpoint(seq SeqID) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.checkpointed == nil {
		r.checkpointed = map[SeqID]bool{}
	}
	r.checkpointed[seq] = true
	r.ops = append(r.ops, rewOp{"checkpoint", seq})
	return nil
}

func (r *recordingRewinder) Rollback(seq SeqID, _ Pos) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ops = append(r.ops, rewOp{"rollback", seq})
	return nil
}

func (r *recordingRewinder) DropCheckpoint(seq SeqID) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.checkpointed, seq)
	r.ops = append(r.ops, rewOp{"drop", seq})
}
