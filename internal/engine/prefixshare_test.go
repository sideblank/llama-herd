// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package engine

import (
	"context"
	"errors"
	"testing"
)

type fakeSharer struct {
	cached   map[SeqID]int // highest position held, -1 for none
	calls    []string
	fail     error
	noEffect bool
}

func newSharer() *fakeSharer { return &fakeSharer{cached: map[SeqID]int{}} }

func (f *fakeSharer) SharePrefix(src, dst SeqID, n int) error {
	if f.fail != nil {
		return f.fail
	}
	f.calls = append(f.calls, string(rune('0'+src))+"->"+string(rune('0'+dst)))
	if !f.noEffect {
		f.cached[dst] = n - 1
	}
	return nil
}

func (f *fakeSharer) CachedThrough(seq SeqID) int {
	if p, ok := f.cached[seq]; ok {
		return p
	}
	return -1
}

// sharingFake is a backend that can share prefixes: the standard fake plus the optional
// capability, which is exactly the shape a real backend takes.
type sharingFake struct {
	*fakeBackend
	*fakeSharer
}

// engineWith builds an Engine whose backend can share, without booting a model.
func engineWith(s *fakeSharer) *Engine {
	return &Engine{be: sharingFake{fakeBackend: newFake(8, 512), fakeSharer: s}}
}

func toks(vals ...int) []Token {
	out := make([]Token, len(vals))
	for i, v := range vals {
		out[i] = Token(v)
	}
	return out
}

// hdr builds a prompt of n identical leading tokens followed by a distinct tail.
func hdr(n int, tail ...int) []Token {
	out := make([]Token, 0, n+len(tail))
	for i := 0; i < n; i++ {
		out = append(out, Token(7))
	}
	for _, t := range tail {
		out = append(out, Token(t))
	}
	return out
}

func freshSlot(seq SeqID, prompt []Token) *slot {
	return &slot{seq: seq, promptToks: prompt, pending: prompt, ctx: context.Background()}
}

func residentSlot(seq SeqID, prompt []Token, cachedThrough int) *slot {
	return &slot{seq: seq, promptToks: prompt, pos: Pos(cachedThrough), primed: true,
		ctx: context.Background()}
}

func TestFreshSlotAdoptsAResidentPrefix(t *testing.T) {
	f := newSharer()
	e := engineWith(f)
	donor := residentSlot(0, hdr(1000, 1), 1001)
	newer := freshSlot(1, hdr(1000, 2))
	e.sharePrefixes(map[SeqID]*slot{0: donor, 1: newer})

	if newer.sharedPrefix != 1000 {
		t.Fatalf("want a 1000-token prefix adopted, got %d", newer.sharedPrefix)
	}
	if newer.pos != 1000 {
		t.Fatalf("position must advance past the shared run, got %d", newer.pos)
	}
	if len(newer.pending) != 1 || newer.pending[0] != Token(2) {
		t.Fatalf("only the distinct tail should remain to prefill, got %v", newer.pending)
	}
	if e.c.prefixTokensSaved.Load() != 1000 {
		t.Fatalf("saved prefill must be counted, got %d", e.c.prefixTokensSaved.Load())
	}
}

// The bug this guards: llama produces logits from tokens given this pass.
func TestABorrowerAlwaysKeepsATokenToFeed(t *testing.T) {
	f := newSharer()
	e := engineWith(f)
	same := hdr(1000)
	donor := residentSlot(0, same, 1000)
	newer := freshSlot(1, same) // identical prompt
	e.sharePrefixes(map[SeqID]*slot{0: donor, 1: newer})

	if len(newer.pending) == 0 {
		t.Fatal("a prompt entirely absorbed into a shared prefix leaves nothing to sample from")
	}
	if newer.sharedPrefix != len(same)-1 {
		t.Fatalf("want %d shared, got %d", len(same)-1, newer.sharedPrefix)
	}
}

// The donor must actually hold the cells.
func TestNoSharingFromASlotThatHasNotPrefilledYet(t *testing.T) {
	f := newSharer()
	e := engineWith(f)
	a := freshSlot(0, hdr(1000, 1)) // admitted this tick, holds nothing
	b := freshSlot(1, hdr(1000, 2))
	e.sharePrefixes(map[SeqID]*slot{0: a, 1: b})

	if b.sharedPrefix != 0 || len(f.calls) != 0 {
		t.Fatal("copying from a sequence with no cells would share nothing while reporting success")
	}
}

func TestSharingIsCappedAtWhatTheDonorHasCached(t *testing.T) {
	f := newSharer()
	e := engineWith(f)
	// Donor's prompt is long but only 300 positions are cached so far.
	donor := residentSlot(0, hdr(1000, 1), 300)
	donor.primed = false
	newer := freshSlot(1, hdr(1000, 2))
	e.sharePrefixes(map[SeqID]*slot{0: donor, 1: newer})

	if newer.sharedPrefix != 300 {
		t.Fatalf("cannot share cells the donor has not computed: want 300, got %d", newer.sharedPrefix)
	}
}

func TestShortPrefixesAreNotWorthSharing(t *testing.T) {
	f := newSharer()
	e := engineWith(f)
	donor := residentSlot(0, hdr(10, 1), 11)
	newer := freshSlot(1, hdr(10, 2))
	e.sharePrefixes(map[SeqID]*slot{0: donor, 1: newer})

	if newer.sharedPrefix != 0 {
		t.Fatal("re-tagging every cell has a cost; below the threshold it is not worth paying")
	}
}

// ⛔ The trap the verification exists for.
func TestAShareThatReportsSuccessAndDoesNothingIsCaught(t *testing.T) {
	f := newSharer()
	f.noEffect = true
	e := engineWith(f)
	donor := residentSlot(0, hdr(1000, 1), 1001)
	newer := freshSlot(1, hdr(1000, 2))
	e.sharePrefixes(map[SeqID]*slot{0: donor, 1: newer})

	if newer.sharedPrefix != 0 || newer.pos != 0 {
		t.Fatal("an unverified share leaves a slot generating fluently from an empty history")
	}
	if len(newer.pending) != 1001 {
		t.Fatalf("the slot must fall back to prefilling everything, got %d pending", len(newer.pending))
	}
	if e.c.prefixShareFailed.Load() != 1 {
		t.Fatal("a failed share degrades to ordinary prefill and is invisible unless counted")
	}
}

func TestAFailedShareFallsBackCleanly(t *testing.T) {
	f := newSharer()
	f.fail = errors.New("cache full")
	e := engineWith(f)
	donor := residentSlot(0, hdr(1000, 1), 1001)
	newer := freshSlot(1, hdr(1000, 2))
	e.sharePrefixes(map[SeqID]*slot{0: donor, 1: newer})

	if newer.pos != 0 || len(newer.pending) != 1001 {
		t.Fatal("nothing may be half-applied: pos and pending stay untouched until the share verifies")
	}
	if e.c.prefixShareFailed.Load() != 1 {
		t.Fatal("want the failure counted")
	}
}

func TestNoSharingWithoutACommonPrefix(t *testing.T) {
	f := newSharer()
	e := engineWith(f)
	donor := residentSlot(0, toks(1, 2, 3), 3)
	newer := freshSlot(1, toks(9, 8, 7))
	e.sharePrefixes(map[SeqID]*slot{0: donor, 1: newer})
	if newer.sharedPrefix != 0 || len(f.calls) != 0 {
		t.Fatal("unrelated prompts share nothing")
	}
}

func TestAlreadyPrimedSlotsAreLeftAlone(t *testing.T) {
	f := newSharer()
	e := engineWith(f)
	donor := residentSlot(0, hdr(1000, 1), 1001)
	other := residentSlot(1, hdr(1000, 2), 500) // mid-flight
	e.sharePrefixes(map[SeqID]*slot{0: donor, 1: other})
	if other.sharedPrefix != 0 {
		t.Fatal("sharing into a partly-filled sequence would duplicate positions it already holds")
	}
}

func TestTheLongestMatchWins(t *testing.T) {
	f := newSharer()
	e := engineWith(f)
	short := residentSlot(0, hdr(300, 1), 301)
	long := residentSlot(1, hdr(900, 2), 901)
	newer := freshSlot(2, hdr(900, 3))
	e.sharePrefixes(map[SeqID]*slot{0: short, 1: long, 2: newer})
	if newer.sharedPrefix != 900 {
		t.Fatalf("want the 900-token donor, got %d", newer.sharedPrefix)
	}
}

func TestDonorChoiceIsDeterministic(t *testing.T) {
	// Two donors offering an equal prefix. Go randomises map iteration, so an unstable choice
	// would make identical inputs produce different plans.
	first := -1
	for i := 0; i < 40; i++ {
		f := newSharer()
		e := engineWith(f)
		active := map[SeqID]*slot{
			0: residentSlot(0, hdr(900, 1), 901),
			1: residentSlot(1, hdr(900, 2), 901),
			2: freshSlot(2, hdr(900, 3)),
		}
		e.sharePrefixes(active)
		if len(f.calls) != 1 {
			t.Fatalf("want one share, got %v", f.calls)
		}
		got := int(f.calls[0][0])
		if first == -1 {
			first = got
			continue
		}
		if got != first {
			t.Fatal("donor choice must not depend on map iteration order")
		}
	}
}

func TestBackendWithoutSharingIsANoop(t *testing.T) {
	e := &Engine{be: newFake(8, 512)}
	newer := freshSlot(1, hdr(1000, 2))
	donor := residentSlot(0, hdr(1000, 1), 1001)
	e.sharePrefixes(map[SeqID]*slot{0: donor, 1: newer})
	if newer.sharedPrefix != 0 {
		t.Fatal("a backend that cannot share must simply prefill normally")
	}
}

func TestCommonPrefixOnTokens(t *testing.T) {
	if n := commonPrefix(toks(1, 2, 3, 4), toks(1, 2, 9)); n != 2 {
		t.Fatalf("want 2, got %d", n)
	}
	if n := commonPrefix(nil, toks(1)); n != 0 {
		t.Fatalf("want 0, got %d", n)
	}
}

func TestSharingIsVisibleInStats(t *testing.T) {
	f := newSharer()
	e := engineWith(f)
	donor := residentSlot(0, hdr(1000, 1), 1001)
	newer := freshSlot(1, hdr(1000, 2))
	e.sharePrefixes(map[SeqID]*slot{0: donor, 1: newer})

	st := e.Stats()
	if st.PrefixSharedTotal != 1 || st.PrefixTokensSaved != 1000 {
		t.Fatalf("a saving nobody can read cannot be verified: %+v", st)
	}
}

func TestAFailedShareIsVisibleInStats(t *testing.T) {
	f := newSharer()
	f.noEffect = true
	e := engineWith(f)
	donor := residentSlot(0, hdr(1000, 1), 1001)
	newer := freshSlot(1, hdr(1000, 2))
	e.sharePrefixes(map[SeqID]*slot{0: donor, 1: newer})

	if st := e.Stats(); st.PrefixShareFailed != 1 {
		t.Fatal("a failed share degrades to ordinary prefill and is indistinguishable from " +
			"never having tried — without this counter a broken sharing path looks healthy")
	}
}
