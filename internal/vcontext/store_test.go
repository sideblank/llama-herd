// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestStoreKeepsChunkOrder(t *testing.T) {
	s := NewMemoryStore()
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		if err := s.Put(ctx, "req1", fmt.Sprintf("chunk%d", i),
			Entry{Chunk: i, Text: fmt.Sprintf("piece %d", i)}); err != nil {
			t.Fatal(err)
		}
	}
	all, err := s.All(ctx, "req1")
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 5 {
		t.Fatalf("got %d entries, want 5", len(all))
	}
	for i, e := range all {
		if e.Chunk != i {
			t.Errorf("position %d holds chunk %d — reassembly depends on this order", i, e.Chunk)
		}
	}
}

// Sessions must not see each other. Two concurrent requests sharing a store would otherwise
// reassemble each other's chunks, which produces a plausible answer to neither.
func TestSessionsAreIsolated(t *testing.T) {
	s := NewMemoryStore()
	ctx := context.Background()
	_ = s.Put(ctx, "a", "k", Entry{Text: "from a"})
	_ = s.Put(ctx, "b", "k", Entry{Text: "from b"})

	got, ok, _ := s.Get(ctx, "a", "k")
	if !ok || got.Text != "from a" {
		t.Errorf("session a sees %q", got.Text)
	}
	got, ok, _ = s.Get(ctx, "b", "k")
	if !ok || got.Text != "from b" {
		t.Errorf("session b sees %q", got.Text)
	}
}

// Close is the normal cleanup path and must actually release the state.
func TestCloseReleasesTheSession(t *testing.T) {
	s := NewMemoryStore()
	ctx := context.Background()
	_ = s.Put(ctx, "req", "k", Entry{Text: "x"})
	if s.Sessions() != 1 {
		t.Fatalf("expected 1 session, got %d", s.Sessions())
	}
	if err := s.Close(ctx, "req"); err != nil {
		t.Fatal(err)
	}
	if s.Sessions() != 0 {
		t.Errorf("session survived Close — this is the primary cleanup path")
	}
	if _, ok, _ := s.Get(ctx, "req", "k"); ok {
		t.Error("closed session still returns data")
	}
}

// Expiry is the backstop, and it is the one that matters: Close is skipped exactly when things go
// wrong — a cancelled request, a panic, a caller that disappears mid-stream. State that only
// cleans up on the happy path is a leak with extra steps.
func TestAbandonedSessionsExpire(t *testing.T) {
	s := NewMemoryStore()
	s.TTL = time.Minute
	now := time.Now()
	s.now = func() time.Time { return now }
	ctx := context.Background()

	_ = s.Put(ctx, "abandoned", "k", Entry{Text: "never closed"})
	if s.Sessions() != 1 {
		t.Fatalf("expected the session to exist, got %d", s.Sessions())
	}

	now = now.Add(2 * time.Minute) // the caller went away and never called Close
	if s.Sessions() != 0 {
		t.Errorf("abandoned session survived its TTL — %d still held", s.Sessions())
	}
	if _, ok, _ := s.Get(ctx, "abandoned", "k"); ok {
		t.Error("expired session still returns data")
	}
}

// Activity must keep a live session alive. A long request whose chunks are still reporting must
// not have its state expire underneath it.
func TestActiveSessionsSurvive(t *testing.T) {
	s := NewMemoryStore()
	s.TTL = time.Minute
	now := time.Now()
	s.now = func() time.Time { return now }
	ctx := context.Background()

	_ = s.Put(ctx, "busy", "k0", Entry{})
	for i := 1; i < 5; i++ {
		now = now.Add(40 * time.Second) // less than the TTL each time, more in total
		if err := s.Put(ctx, "busy", fmt.Sprintf("k%d", i), Entry{}); err != nil {
			t.Fatal(err)
		}
		if s.Sessions() != 1 {
			t.Fatalf("a session in active use expired at step %d", i)
		}
	}
	all, _ := s.All(ctx, "busy")
	if len(all) != 5 {
		t.Errorf("got %d entries, want 5 — earlier chunks were dropped from a live session",
			len(all))
	}
}

// Chunks run in parallel, which is the whole point, so the store has to take that.
func TestConcurrentChunksAreSafe(t *testing.T) {
	s := NewMemoryStore()
	ctx := context.Background()
	var wg sync.WaitGroup
	for i := 0; i < 48; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_ = s.Put(ctx, "req", fmt.Sprintf("chunk%d", i), Entry{Chunk: i})
			_, _, _ = s.Get(ctx, "req", "chunk0")
			_, _ = s.All(ctx, "req")
		}(i)
	}
	wg.Wait()
	all, _ := s.All(ctx, "req")
	if len(all) != 48 {
		t.Errorf("got %d entries from 48 concurrent chunks", len(all))
	}
}

func TestPutRequiresSessionAndKey(t *testing.T) {
	s := NewMemoryStore()
	ctx := context.Background()
	if err := s.Put(ctx, "", "k", Entry{}); err == nil {
		t.Error("expected an error for an empty session")
	}
	if err := s.Put(ctx, "s", "", Entry{}); err == nil {
		t.Error("expected an error for an empty key")
	}
}

// MemoryStore must satisfy the interface a real backend would replace it through.
var _ Store = (*MemoryStore)(nil)

// A TTL bounds how long state lives; it does not bound how much exists at once. Under a burst of
// requests that all abandon, the store must still be bounded.
func TestSessionCountIsCappedRegardlessOfTTL(t *testing.T) {
	s := NewMemoryStore()
	s.MaxSessions = 10
	s.TTL = time.Hour // nothing expires during this test
	now := time.Now()
	s.now = func() time.Time { return now }
	ctx := context.Background()

	for i := 0; i < 100; i++ {
		now = now.Add(time.Second) // each session newer than the last
		if err := s.Put(ctx, fmt.Sprintf("req%d", i), "k", Entry{}); err != nil {
			t.Fatal(err)
		}
	}
	if got := s.Sessions(); got > 10 {
		t.Errorf("holding %d sessions against a cap of 10 — a TTL alone does not bound a burst",
			got)
	}
	if s.Evicted() == 0 {
		t.Error("sessions were dropped by the cap but Evicted() reports none")
	}
	// The most recently touched must survive; the oldest are the likely-abandoned ones.
	if _, ok, _ := s.Get(ctx, "req99", "k"); !ok {
		t.Error("the newest session was evicted — eviction must drop least-recently-touched")
	}
}

// One session must not grow without end if a caller writes keys in a loop.
func TestSessionEntriesAreCapped(t *testing.T) {
	s := NewMemoryStore()
	s.MaxEntriesPerSession = 8
	ctx := context.Background()

	var lastErr error
	for i := 0; i < 50; i++ {
		if err := s.Put(ctx, "req", fmt.Sprintf("k%d", i), Entry{}); err != nil {
			lastErr = err
			break
		}
	}
	if lastErr == nil {
		t.Fatal("expected the entry cap to refuse further growth")
	}
	all, _ := s.All(ctx, "req")
	if len(all) > 8 {
		t.Errorf("session holds %d entries against a cap of 8", len(all))
	}
}

// Overwriting an existing key must not count against the cap or grow the order list — a chunk
// revising its own contribution is normal.
func TestOverwriteDoesNotGrowTheSession(t *testing.T) {
	s := NewMemoryStore()
	s.MaxEntriesPerSession = 4
	ctx := context.Background()
	for i := 0; i < 100; i++ {
		if err := s.Put(ctx, "req", "same-key", Entry{Chunk: i}); err != nil {
			t.Fatalf("overwrite %d refused: %v", i, err)
		}
	}
	all, _ := s.All(ctx, "req")
	if len(all) != 1 {
		t.Errorf("100 overwrites produced %d entries, want 1", len(all))
	}
	if all[0].Chunk != 99 {
		t.Errorf("overwrite did not take: chunk = %d", all[0].Chunk)
	}
}
