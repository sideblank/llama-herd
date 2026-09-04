// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package prefix

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

func toks(v ...int32) []Token { return v }

func TestSharedHeaderIsFound(t *testing.T) {
	hdr := toks(1, 2, 3, 4, 5)
	var prompts [][]Token
	for i := 0; i < 48; i++ {
		p := append(append([]Token{}, hdr...), Token(100+i), Token(200+i))
		prompts = append(prompts, p)
	}
	plan, err := Analyse(prompts)
	if err != nil {
		t.Fatal(err)
	}
	if plan.SharedLen() != 5 {
		t.Fatalf("want a 5-token shared header, got %d", plan.SharedLen())
	}
	if plan.Saved != 5*47 {
		t.Fatalf("47 sequences avoid the prefill: want %d, got %d", 5*47, plan.Saved)
	}
	for i, s := range plan.Suffixes {
		if len(s) != 2 || s[0] != Token(100+i) {
			t.Fatalf("suffix %d wrong: %v", i, s)
		}
	}
}

func TestFractionMatchesTheHeaderArithmetic(t *testing.T) {
	// 48 streams, 1500-token header, 8874-token streams: the case that motivates this.
	hdr := make([]Token, 1500)
	for i := range hdr {
		hdr[i] = Token(i)
	}
	var prompts [][]Token
	for i := 0; i < 48; i++ {
		body := make([]Token, 7374)
		for j := range body {
			body[j] = Token(1_000_000 + i*10_000 + j)
		}
		prompts = append(prompts, append(append([]Token{}, hdr...), body...))
	}
	plan, _ := Analyse(prompts)
	if plan.SharedLen() != 1500 {
		t.Fatalf("got %d", plan.SharedLen())
	}
	// 1500*47 = 70,500 of 48*8874 = 425,952 -> ~16.6%
	if f := plan.Fraction(); f < 0.16 || f > 0.17 {
		t.Fatalf("want ~16.6%% of prefill avoided, got %.1f%%", f*100)
	}
}

func TestNoCommonPrefixIsNotAnError(t *testing.T) {
	plan, err := Analyse([][]Token{toks(1, 2), toks(3, 4)})
	if err != nil {
		t.Fatal(err)
	}
	if plan.SharedLen() != 0 || plan.Saved != 0 {
		t.Fatalf("nothing is shared: %+v", plan)
	}
	if plan.Worth(1) {
		t.Fatal("a plan sharing nothing is never worth executing")
	}
}

// The bug this guards: a sequence whose entire prompt is absorbed has nothing left to evaluate.
func TestAPromptIsNeverFullyAbsorbedIntoThePrefix(t *testing.T) {
	prompts := [][]Token{toks(1, 2, 3), toks(1, 2, 3, 4, 5)}
	plan, err := Analyse(prompts)
	if err != nil {
		t.Fatal(err)
	}
	if plan.SharedLen() != 2 {
		t.Fatalf("the shared run must stop one short of the shortest prompt, got %d", plan.SharedLen())
	}
	for i, s := range plan.Suffixes {
		if len(s) == 0 {
			t.Fatalf("suffix %d is empty — llama produces logits from tokens given this pass, so "+
				"a sequence with nothing to evaluate has nothing to sample from", i)
		}
	}
}

func TestIdenticalPromptsStillLeaveATokenEach(t *testing.T) {
	p := toks(7, 7, 7, 7)
	plan, _ := Analyse([][]Token{p, p, p})
	if plan.SharedLen() != 3 {
		t.Fatalf("got %d", plan.SharedLen())
	}
	for _, s := range plan.Suffixes {
		if len(s) != 1 {
			t.Fatalf("every sequence keeps exactly one token, got %d", len(s))
		}
	}
}

func TestSinglePromptSharesNothing(t *testing.T) {
	plan, err := Analyse([][]Token{toks(1, 2, 3)})
	if err != nil {
		t.Fatal(err)
	}
	if plan.SharedLen() != 0 || plan.Worth(1) {
		t.Fatal("there is nobody to share with")
	}
}

func TestEmptyBatchIsAnError(t *testing.T) {
	if _, err := Analyse(nil); !errors.Is(err, ErrNoPrompts) {
		t.Fatalf("got %v", err)
	}
}

func TestPrefixDoesNotAliasTheCallersSlice(t *testing.T) {
	a := toks(1, 2, 3, 9)
	b := toks(1, 2, 3, 8)
	plan, _ := Analyse([][]Token{a, b})
	plan.Prefix = append(plan.Prefix, 42)
	if a[3] == 42 {
		t.Fatal("appending to the plan's prefix wrote into the caller's prompt")
	}
}

// --- sharing ---

type fakeCache struct {
	unified  bool
	pos      map[int]int
	decoded  []string
	copies   []string
	copyNoop bool
	failCopy error
}

func newCache() *fakeCache { return &fakeCache{unified: true, pos: map[int]int{}} }

func (f *fakeCache) Unified() bool { return f.unified }
func (f *fakeCache) DecodeSeq(seq int, tokens []Token, from int) error {
	f.decoded = append(f.decoded, fmt.Sprintf("seq%d+%d@%d", seq, len(tokens), from))
	f.pos[seq] = from + len(tokens) - 1
	return nil
}
func (f *fakeCache) CopyPrefix(src, dst, n int) error {
	if f.failCopy != nil {
		return f.failCopy
	}
	f.copies = append(f.copies, fmt.Sprintf("%d->%d:%d", src, dst, n))
	if f.copyNoop {
		return nil // reports success, changes nothing — upstream's silent path
	}
	f.pos[dst] = n - 1
	return nil
}
func (f *fakeCache) PosMax(seq int) int {
	if p, ok := f.pos[seq]; ok {
		return p
	}
	return -1
}

func TestShareComputesOnceAndCopiesToTheRest(t *testing.T) {
	hdr := toks(1, 2, 3, 4)
	prompts := [][]Token{
		append(append([]Token{}, hdr...), 10),
		append(append([]Token{}, hdr...), 20),
		append(append([]Token{}, hdr...), 30),
	}
	plan, _ := Analyse(prompts)
	c := newCache()
	if err := Share(c, plan, []int{0, 1, 2}); err != nil {
		t.Fatal(err)
	}
	if len(c.decoded) != 1 {
		t.Fatalf("the prefix must be computed exactly once, got %v", c.decoded)
	}
	if len(c.copies) != 2 {
		t.Fatalf("two destinations, got %v", c.copies)
	}
}

// The trap the whole wrapper exists for.
func TestASilentlyIneffectiveCopyIsCaught(t *testing.T) {
	prompts := [][]Token{toks(1, 2, 3, 10), toks(1, 2, 3, 20)}
	plan, _ := Analyse(prompts)
	c := newCache()
	c.copyNoop = true
	err := Share(c, plan, []int{0, 1})
	if err == nil {
		t.Fatal("llama_memory_seq_cp returns void and has a path that does nothing; an unverified " +
			"copy leaves a sequence generating fluently from an empty history, which surfaces as a " +
			"bad answer rather than an error")
	}
	if !strings.Contains(err.Error(), "did nothing") {
		t.Fatalf("the error should say what happened: %v", err)
	}
}

func TestSharingRefusesWithoutAUnifiedCache(t *testing.T) {
	plan, _ := Analyse([][]Token{toks(1, 2, 3, 4), toks(1, 2, 3, 5)})
	c := newCache()
	c.unified = false
	if err := Share(c, plan, []int{0, 1}); !errors.Is(err, ErrNotUnified) {
		t.Fatalf("upstream asserts on a partial cross-stream copy and takes the process down, so "+
			"this must be refused here; got %v", err)
	}
	if len(c.decoded) != 0 {
		t.Fatal("nothing should have been decoded before the refusal")
	}
}

func TestSharingNothingIsANoop(t *testing.T) {
	plan, _ := Analyse([][]Token{toks(1, 2), toks(3, 4)})
	c := newCache()
	if err := Share(c, plan, []int{0, 1}); err != nil {
		t.Fatal(err)
	}
	if len(c.decoded) != 0 || len(c.copies) != 0 {
		t.Fatal("no shared prefix means no work")
	}
}

func TestSeqIDCountMustMatch(t *testing.T) {
	plan, _ := Analyse([][]Token{toks(1, 2, 3), toks(1, 2, 4)})
	if err := Share(newCache(), plan, []int{0}); err == nil {
		t.Fatal("a short sequence-id list would silently leave prompts unassigned")
	}
}

func TestCopyFailureIsSurfaced(t *testing.T) {
	plan, _ := Analyse([][]Token{toks(1, 2, 3, 4), toks(1, 2, 3, 5)})
	c := newCache()
	c.failCopy = errors.New("cache full")
	if err := Share(c, plan, []int{0, 1}); err == nil || !strings.Contains(err.Error(), "cache full") {
		t.Fatalf("got %v", err)
	}
}

func TestWorthThreshold(t *testing.T) {
	plan, _ := Analyse([][]Token{toks(1, 2, 3, 10), toks(1, 2, 3, 20)})
	if !plan.Worth(3) {
		t.Fatal("a 3-token prefix clears a threshold of 3")
	}
	if plan.Worth(4) {
		t.Fatal("re-tagging every cell has a cost; below the threshold it is not worth paying")
	}
}
