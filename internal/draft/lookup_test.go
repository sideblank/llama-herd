// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package draft

import (
	"testing"

	"github.com/sideblank/llama-herd/internal/engine"
)

func toks(s string) []engine.Token {
	out := make([]engine.Token, 0, len(s))
	for _, r := range s {
		out = append(out, engine.Token(r))
	}
	return out
}

func str(t []engine.Token) string {
	b := make([]rune, len(t))
	for i, x := range t {
		b[i] = rune(x)
	}
	return string(b)
}

func TestProposesTheContinuationThatFollowedBefore(t *testing.T) {
	l := NewLookup(4)
	// The pattern "abc" appeared once already, followed by "xyz".
	l.Seed(0, toks("abcxyz-----ab"))

	got, err := l.Draft(0, 'c', 0, 4)
	if err != nil {
		t.Fatal(err)
	}
	if str(got) != "xyz-" {
		t.Fatalf("proposed %q, want %q", str(got), "xyz-")
	}
}

func TestProposesNothingWithoutAMatch(t *testing.T) {
	l := NewLookup(4)
	l.Seed(0, toks("the quick brown fox"))
	got, _ := l.Draft(0, 'z', 0, 4)
	if len(got) != 0 {
		t.Fatalf("proposed %q against no matching history", str(got))
	}
}

// The pattern must not match its own position, or every step would propose the tokens that
// already follow it and speculation would be a no-op dressed as a hit.
func TestPatternDoesNotMatchItself(t *testing.T) {
	l := NewLookup(4)
	l.Seed(0, toks("xyzabc"))
	got, _ := l.Draft(0, 'd', 0, 4)
	if len(got) != 0 {
		t.Fatalf("proposed %q from a self-match", str(got))
	}
}

// Recency predicts better: when a pattern occurs twice, the nearer continuation wins.
func TestPrefersTheMostRecentOccurrence(t *testing.T) {
	l := NewLookup(4)
	//        old continuation "111"        recent continuation "222"
	l.Seed(0, toks("keyOLD111................keyNEW222................ke"))
	// Search on pattern "key" — the nearest earlier "key" is followed by "NEW".
	got, _ := l.Draft(0, 'y', 0, 3)
	if str(got) != "NEW" {
		t.Fatalf("proposed %q, want the most recent continuation %q", str(got), "NEW")
	}
}

func TestRespectsRequestedCount(t *testing.T) {
	l := NewLookup(8)
	l.Seed(0, toks("abcdefghij--------ab"))
	got, _ := l.Draft(0, 'c', 0, 2)
	if len(got) != 2 {
		t.Fatalf("proposed %d tokens, want 2", len(got))
	}
	if str(got) != "de" {
		t.Fatalf("proposed %q, want %q", str(got), "de")
	}
}

func TestReleaseDropsHistory(t *testing.T) {
	l := NewLookup(4)
	l.Seed(0, toks("abcxyz----abc"))
	l.Release(0)
	got, _ := l.Draft(0, 'c', 0, 4)
	if len(got) != 0 {
		t.Fatalf("history survived release: proposed %q", str(got))
	}
}

// Sequences must not see each other's history — the same failure mode as a shared sampler.
func TestHistoryIsPerSequence(t *testing.T) {
	l := NewLookup(4)
	l.Seed(0, toks("abcxyz----ab"))
	l.Seed(1, toks("nothing relevant here"))

	got0, _ := l.Draft(0, 'c', 0, 3)
	if str(got0) != "xyz" {
		t.Fatalf("seq 0 proposed %q", str(got0))
	}
	got1, _ := l.Draft(1, 'c', 0, 3)
	if len(got1) != 0 {
		t.Fatalf("seq 1 saw seq 0's history: %q", str(got1))
	}
}

func TestHistoryIsBounded(t *testing.T) {
	l := NewLookup(4)
	l.MaxHistory = 64
	big := make([]engine.Token, 10000)
	for i := range big {
		big[i] = engine.Token('a' + i%26)
	}
	l.Seed(0, big)

	l.mu.Lock()
	n := len(l.hist[0])
	l.mu.Unlock()
	if n > 64 {
		t.Fatalf("history is %d tokens against a bound of 64", n)
	}
}
