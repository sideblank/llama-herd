// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package engine

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestRegistryRoutesByName(t *testing.T) {
	r := NewRegistry()
	a, b := newFake(1, 8), newFake(1, 8)
	if err := r.Add("alpha", New(a, Config{}), nil); err != nil {
		t.Fatal(err)
	}
	if err := r.Add("beta", New(b, Config{}), nil); err != nil {
		t.Fatal(err)
	}

	if got := r.Names(); len(got) != 2 || got[0] != "alpha" || got[1] != "beta" {
		t.Fatalf("Names() = %v, want [alpha beta]", got)
	}
	if _, err := r.Get("alpha"); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Get("missing"); !errors.Is(err, ErrUnknownModel) {
		t.Fatalf("want ErrUnknownModel, got %v", err)
	}
}

func TestRegistryRefusesDuplicateName(t *testing.T) {
	r := NewRegistry()
	if err := r.Add("m", New(newFake(1, 8), Config{}), nil); err != nil {
		t.Fatal(err)
	}
	err := r.Add("m", New(newFake(1, 8), Config{}), nil)
	if err == nil || !strings.Contains(err.Error(), "already registered") {
		t.Fatalf("duplicate name should be refused, got %v", err)
	}
}

// A dead engine must be refused rather than accepting requests that would queue forever
// against a decode loop that will never tick again.
func TestDeadEngineIsRefused(t *testing.T) {
	f := newFake(1, 0) // zero batch capacity makes the loop fail on its first tick
	f.script[0] = []Token{'a'}
	e := New(f, Config{})

	r := NewRegistry()
	if err := r.Add("doomed", e, nil); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r.Start(ctx)

	// Submitting wakes the loop, which then dies on the zero-capacity batch.
	if _, err := e.Submit(context.Background(), Request{Prompt: "x"}); err != nil {
		t.Fatal(err)
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := r.Get("doomed"); errors.Is(err, ErrModelDead) {
			if h := r.Health()["doomed"]; h == nil {
				t.Fatal("Health() should report the failure")
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("registry never marked the dead engine")
}

func TestRegistryStartAndCancel(t *testing.T) {
	r := NewRegistry()
	if err := r.Add("m", New(newFake(1, 8), Config{}), nil); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	r.Start(ctx)
	cancel()

	done := make(chan struct{})
	go func() { r.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Wait did not return after cancellation")
	}

	// Cancellation is not a failure: the model should not be marked dead.
	if err := r.Health()["m"]; err != nil {
		t.Fatalf("cancellation marked the model dead: %v", err)
	}
}
