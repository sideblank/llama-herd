// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package draft_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/sideblank/llama-herd/internal/draft"
	"github.com/sideblank/llama-herd/internal/engine"
	"github.com/sideblank/llama-herd/internal/enginetest"
)

// The end-to-end path: the engine seeds the drafter with the prompt, the drafter proposes
// from it, and the engine verifies and accepts. A model echoing its context is the case
// lookup drafting exists for — editing a file that is in the prompt, filling a schema,
// continuing a transcript.
func TestLookupDraftsAcceptedThroughTheEngine(t *testing.T) {
	const phrase = "alpha bravo charlie delta echo "
	be := enginetest.New(1, 256, strings.Repeat(phrase, 6))

	lk := draft.NewLookup(4)
	e := engine.New(be, engine.Config{Drafter: lk})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _ = e.Run(ctx) }()

	s, err := e.Submit(context.Background(), engine.Request{
		Prompt:    strings.Repeat(phrase, 4),
		MaxTokens: 120,
	})
	if err != nil {
		t.Fatal(err)
	}

	deadline := time.After(10 * time.Second)
	for done := false; !done; {
		select {
		case ev, ok := <-s.Events:
			if !ok || ev.Done {
				done = true
			}
		case <-deadline:
			t.Fatal("timed out")
		}
	}

	st := e.Stats()
	if st.DraftsProposed == 0 {
		t.Fatal("nothing proposed — the prompt repeats the generated phrase, so the " +
			"drafter should have found matches; the engine may not be seeding it")
	}
	if st.DraftsAccepted == 0 {
		t.Fatalf("proposed %d and accepted none, though the target emits the same phrase "+
			"the prompt contains", st.DraftsProposed)
	}
	t.Logf("proposed=%d accepted=%d rate=%.0f%% tokens=%d",
		st.DraftsProposed, st.DraftsAccepted, st.AcceptanceRate()*100, st.TokensGenerated)
}
