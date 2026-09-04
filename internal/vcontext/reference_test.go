// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import "testing"

// The planner must be configurable from measurements rather than from guesses, and the two must
// not drift: a change to the measured configuration has to move the planner with it.
func TestPolicyComesFromMeasurements(t *testing.T) {
	pol, ok := PolicyForCard("3090", ShapeReadHeavy)
	if !ok {
		t.Fatal("no policy for the 3090, which has measurements")
	}
	if pol.MaxChunks != 48 {
		t.Errorf("max chunks = %d, want the 48 streams that were measured", pol.MaxChunks)
	}
	if pol.PerStreamContext != 8874 {
		t.Errorf("per-stream context = %d, want the measured 8874", pol.PerStreamContext)
	}
	if err := pol.Validate(); err != nil {
		t.Errorf("a policy built from measurements must be usable: %v", err)
	}
	if pol.usable() <= 0 {
		t.Error("no room left for input")
	}
}

// A card with no measurements must produce no policy. Planning from projections yields plans
// nothing has ever run.
func TestUnmeasuredCardHasNoPolicy(t *testing.T) {
	for _, card := range []string{"4090", "5090", "nonsense"} {
		if _, ok := PolicyForCard(card, ShapeUnknown); ok {
			t.Errorf("%s has no measurements and must not yield a policy", card)
		}
	}
}

// End to end on the real profile: a 131k input, read-heavy, must split into chunks that fit.
func TestMeasuredProfilePlansARealRequest(t *testing.T) {
	pol, _ := PolicyForCard("3090", ShapeReadHeavy)
	plan, err := (Planner{Policy: pol}).Plan(131072)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Refused {
		t.Fatalf("131k refused on a 48-stream deployment: %s", plan.Reason)
	}
	if plan.ChunkTokens > pol.usable() {
		t.Errorf("chunk of %d exceeds %d usable", plan.ChunkTokens, pol.usable())
	}
	t.Logf("131k read-heavy on a 3090: %d chunks of %d — %s",
		plan.Chunks, plan.ChunkTokens, plan.Reason)
}
