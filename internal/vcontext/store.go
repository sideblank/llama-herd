// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Store holds what chunks need to tell each other, for as long as one request lives.
//
// Chunks cannot see each other — that is the cost of splitting — so anything relating them has to
// live outside the model. This is that place.
//
// It is an interface because the right implementation depends on a product decision that has not
// been made: whether cross-chunk state must outlive a request. If it does not, an in-process map
// is the correct answer and a database is a dependency bought for nothing. If it does — a document
// session a caller returns to, or state shared across replicas — this interface is where a real
// store goes.
//
// Implementations must be safe for concurrent use: chunks run in parallel, which is the point.
type Store interface {
	// Put records a chunk's contribution. Overwrites any previous value for the same key.
	Put(ctx context.Context, session, key string, value Entry) error
	// Get returns one entry, and whether it was there.
	Get(ctx context.Context, session, key string) (Entry, bool, error)
	// All returns every entry for a session, in insertion order, which is chunk order.
	All(ctx context.Context, session string) ([]Entry, error)
	// Close releases a session's state. Called when a request finishes, successfully or not.
	//
	// This is the primary cleanup path; expiry is the backstop for when it is not reached.
	Close(ctx context.Context, session string) error
}

// Entry is one chunk's contribution to a request.
type Entry struct {
	// Chunk is which piece produced this, so reassembly can order and attribute it.
	Chunk int
	// Text is the chunk's output — a summary, an extraction, whatever the contract asks for.
	Text string
	// Meta carries anything the reassembly contract needs that is not the text itself.
	Meta map[string]string
	// At is when it was recorded.
	At time.Time
}

// MemoryStore keeps session state in process, expiring anything a request forgot to close.
//
// Two cleanup paths on purpose. Close is the normal one and runs when a request ends. Expiry
// exists because the normal path is the one that gets skipped — a cancelled request, a panic, a
// caller that goes away mid-stream — and state that only cleans up on the happy path is a leak
// with extra steps.
type MemoryStore struct {
	// TTL bounds how long an unclosed session survives. Zero uses DefaultTTL.
	TTL time.Duration
	// MaxSessions bounds how many are held at once, whatever the TTL says. Zero uses
	// DefaultMaxSessions.
	//
	// A TTL alone bounds memory by TIME, not by COUNT: a burst of requests that all abandon
	// still accumulates for the whole window. This is the bound that holds under load, and it
	// evicts least-recently-touched first, which is the one most likely to be abandoned.
	MaxSessions int
	// MaxEntriesPerSession bounds one session's growth. Chunk count already bounds this in
	// normal use; the cap is for a caller that writes keys in a loop. Zero uses the default.
	MaxEntriesPerSession int

	mu       sync.Mutex
	sessions map[string]*session
	now      func() time.Time // injectable so expiry is testable without sleeping
	// evicted counts sessions dropped by a cap rather than by Close or TTL. Non-zero means
	// requests are being abandoned faster than they expire, which is worth seeing.
	evicted uint64
}

// DefaultTTL is generous against a request's real lifetime and short against a leak. A chunked
// request is seconds to a couple of minutes; anything still here after this was abandoned.
const DefaultTTL = 15 * time.Minute

// DefaultMaxSessions caps concurrent sessions. Well above any real concurrency for a single
// engine — a deployment runs tens of streams, not thousands of simultaneous chunked requests —
// and low enough that a leak is bounded rather than fatal.
const DefaultMaxSessions = 1024

// DefaultMaxEntriesPerSession bounds one request's state. Chunk count is the real limit and is
// far below this; the cap exists so a caller writing keys in a loop cannot grow a session without
// end.
const DefaultMaxEntriesPerSession = 4096

type session struct {
	order   []string
	entries map[string]Entry
	touched time.Time
}

// NewMemoryStore returns a store that keeps state in process.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{sessions: map[string]*session{}, now: time.Now}
}

func (m *MemoryStore) ttl() time.Duration {
	if m.TTL > 0 {
		return m.TTL
	}
	return DefaultTTL
}

func (m *MemoryStore) maxSessions() int {
	if m.MaxSessions > 0 {
		return m.MaxSessions
	}
	return DefaultMaxSessions
}

func (m *MemoryStore) maxEntries() int {
	if m.MaxEntriesPerSession > 0 {
		return m.MaxEntriesPerSession
	}
	return DefaultMaxEntriesPerSession
}

// sweep drops expired sessions and then enforces the count cap.
//
// Called on every operation rather than from a goroutine: a background sweeper is a lifecycle to
// get wrong — it has to be started, stopped, and not outlive the store — and the cost here is
// proportional to the number of live sessions, which the cap keeps small.
//
// The consequence to be honest about: a store that goes completely idle stops sweeping, so its
// last sessions are held until something touches it again. That is bounded by MaxSessions rather
// than unbounded, and an idle store holding a few kilobytes is not the failure worth adding a
// goroutine lifecycle to prevent.
func (m *MemoryStore) sweep() {
	cutoff := m.now().Add(-m.ttl())
	for id, s := range m.sessions {
		if s.touched.Before(cutoff) {
			delete(m.sessions, id)
		}
	}

	// Then the hard bound. A TTL limits how long state lives; this limits how much exists at
	// once, which is what holds when requests arrive faster than they finish.
	max := m.maxSessions()
	for len(m.sessions) > max {
		oldest, at := "", time.Time{}
		for id, s := range m.sessions {
			if oldest == "" || s.touched.Before(at) {
				oldest, at = id, s.touched
			}
		}
		if oldest == "" {
			break
		}
		delete(m.sessions, oldest)
		m.evicted++
	}
}

func (m *MemoryStore) Put(_ context.Context, sessionID, key string, value Entry) error {
	if sessionID == "" || key == "" {
		return fmt.Errorf("vcontext: session and key are required")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sweep()

	s := m.sessions[sessionID]
	if s == nil {
		s = &session{entries: map[string]Entry{}}
		m.sessions[sessionID] = s
	}
	if _, seen := s.entries[key]; !seen {
		if len(s.order) >= m.maxEntries() {
			return fmt.Errorf("vcontext: session %s already holds %d entries — refusing to "+
				"grow further", sessionID, len(s.order))
		}
		s.order = append(s.order, key)
	}
	if value.At.IsZero() {
		value.At = m.now()
	}
	s.entries[key] = value
	s.touched = m.now()
	return nil
}

func (m *MemoryStore) Get(_ context.Context, sessionID, key string) (Entry, bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sweep()

	s := m.sessions[sessionID]
	if s == nil {
		return Entry{}, false, nil
	}
	e, ok := s.entries[key]
	if ok {
		s.touched = m.now()
	}
	return e, ok, nil
}

func (m *MemoryStore) All(_ context.Context, sessionID string) ([]Entry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sweep()

	s := m.sessions[sessionID]
	if s == nil {
		return nil, nil
	}
	out := make([]Entry, 0, len(s.order))
	for _, k := range s.order {
		out = append(out, s.entries[k])
	}
	s.touched = m.now()
	return out, nil
}

func (m *MemoryStore) Close(_ context.Context, sessionID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.sessions, sessionID)
	m.sweep()
	return nil
}

// Sessions reports how many sessions are held. For tests and for a health endpoint: a number that
// grows without bound is the leak this design is built to avoid.
func (m *MemoryStore) Sessions() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sweep()
	return len(m.sessions)
}

// Evicted counts sessions dropped because the count cap was reached, rather than by Close or by
// expiry. It should stay at zero.
//
// Non-zero means requests are being abandoned faster than they expire — state is being discarded
// while it may still be wanted, which shows up as reassembly missing chunks rather than as an
// error. Worth surfacing on a health endpoint for that reason.
func (m *MemoryStore) Evicted() uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.evicted
}
