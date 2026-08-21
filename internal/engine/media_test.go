// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package engine

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// mediaFake adds the media interface to the package's own fake.
type mediaFake struct {
	*fakeBackend
	marker   string
	calls    int
	lastSeq  SeqID
	lastN    int
	posAfter Pos
	err      error
}

var _ MediaBackend = (*mediaFake)(nil)

func (m *mediaFake) MediaMarker() string { return m.marker }

func (m *mediaFake) PrefillMedia(seq SeqID, _ Pos, _ string, media [][]byte, _ bool) (Pos, error) {
	if m.err != nil {
		return 0, m.err
	}
	m.calls++
	m.lastSeq = seq
	m.lastN = len(media)
	return m.posAfter, nil
}

func newMediaFake() *mediaFake {
	f := newFake(2, 32)
	f.script[0] = []Token{'o', 'k'}
	f.script[1] = []Token{'o', 'k'}
	return &mediaFake{fakeBackend: f, marker: "<img>", posAfter: 40}
}

func TestMediaRequestPrefillsThenDecodesNormally(t *testing.T) {
	m := newMediaFake()
	e := New(m, Config{})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{
		Prompt: "describe <img> please",
		Media:  [][]byte{{1, 2, 3}},
	})
	if err != nil {
		t.Fatal(err)
	}
	text, reason := collect(t, s)

	if m.calls != 1 {
		t.Fatalf("PrefillMedia called %d times, want 1", m.calls)
	}
	if m.lastN != 1 {
		t.Fatalf("prefilled %d media items, want 1", m.lastN)
	}
	if text != "ok" {
		t.Fatalf("text = %q, want %q — decode should continue normally after prefill", text, "ok")
	}
	if reason != ReasonEOS {
		t.Fatalf("reason = %q", reason)
	}
}

// A prompt without the marker means the media is dropped silently and the model answers
// about nothing. That must be an error at submit, not a confusing answer later.
func TestMediaWithoutMarkerIsRejected(t *testing.T) {
	m := newMediaFake()
	e := New(m, Config{})

	_, err := e.Submit(context.Background(), Request{
		Prompt: "describe this please",
		Media:  [][]byte{{1, 2, 3}},
	})
	if err == nil {
		t.Fatal("expected rejection when the prompt lacks the media marker")
	}
	if !strings.Contains(err.Error(), "<img>") {
		t.Fatalf("error should name the required marker, got: %v", err)
	}
}

func TestMediaOnTextOnlyBackendIsRejected(t *testing.T) {
	f := newFake(1, 16) // no MediaBackend
	e := New(f, Config{})

	_, err := e.Submit(context.Background(), Request{Prompt: "x", Media: [][]byte{{1}}})
	if err == nil || !strings.Contains(err.Error(), "cannot accept media") {
		t.Fatalf("want a clear refusal, got: %v", err)
	}
}

func TestMediaPrefillFailureEndsTheStream(t *testing.T) {
	m := newMediaFake()
	m.err = errors.New("projector exploded")
	e := New(m, Config{})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{Prompt: "<img>", Media: [][]byte{{1}}})
	if err != nil {
		t.Fatal(err)
	}

	var gotErr error
	for ev := range s.Events {
		if ev.Err != nil {
			gotErr = ev.Err
		}
	}
	if gotErr == nil || !strings.Contains(gotErr.Error(), "projector exploded") {
		t.Fatalf("failure should surface on the stream, got: %v", gotErr)
	}
}

func TestTextRequestsUnaffectedByMediaSupport(t *testing.T) {
	m := newMediaFake()
	e := New(m, Config{})
	defer run(t, e)()

	s, err := e.Submit(context.Background(), Request{Prompt: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	text, _ := collect(t, s)
	if text != "ok" {
		t.Fatalf("text = %q", text)
	}
	if m.calls != 0 {
		t.Fatalf("PrefillMedia called %d times for a text-only request", m.calls)
	}
}
