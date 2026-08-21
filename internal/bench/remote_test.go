// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// fakeServer stands in for a deployed instance: it streams a fixed number of tokens and
// keeps counters the way the real server does.
type fakeServer struct {
	tokensPerRequest int
	// passesPerRequest models how many forward passes the server spent. Setting it below
	// the token count is what speculative decoding looks like from outside.
	passesPerRequest int

	tokens atomic.Uint64
	passes atomic.Uint64

	mtpLoaded bool
	authSeen  atomic.Value // string
}

func (f *fakeServer) handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /v1/info", func(w http.ResponseWriter, r *http.Request) {
		f.authSeen.Store(r.Header.Get("X-Test-Auth"))
		_ = json.NewEncoder(w).Encode(map[string]any{
			"accelerated": true,
			"devices":     []map[string]any{{"type": "gpu", "free_bytes": 21 << 30}},
			"models": []map[string]any{{
				"name": "m",
				"stats": map[string]any{
					"tokens_generated": f.tokens.Load(),
					"decode_passes":    f.passes.Load(),
					"requests_total":   1,
					"evictions_total":  0,
				},
				"placement": map[string]any{"on_gpu": true, "mtp_loaded": f.mtpLoaded},
			}},
		})
	})

	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			MaxTokens int `json:"max_tokens"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		n := f.tokensPerRequest
		if req.MaxTokens > 0 && req.MaxTokens < n {
			n = req.MaxTokens
		}

		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		for i := 0; i < n; i++ {
			fmt.Fprintf(w, "data: %s\n\n",
				`{"choices":[{"delta":{"content":"x"}}]}`)
			if fl != nil {
				fl.Flush()
			}
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		if fl != nil {
			fl.Flush()
		}

		f.tokens.Add(uint64(n))
		f.passes.Add(uint64(f.passesPerRequest))
	})
	return mux
}

func TestRemoteReadsServerCountersNotClientTiming(t *testing.T) {
	// 60 tokens produced in 30 passes: two tokens per pass, which is what a speculative
	// head landing drafts looks like from outside.
	f := &fakeServer{tokensPerRequest: 60, passesPerRequest: 30, mtpLoaded: true}
	ts := httptest.NewServer(f.handler())
	defer ts.Close()

	r := NewRemote(ts.URL, "", "m", nil)
	res, err := RunRemote(context.Background(), r, Config{
		Model: "m", Prompt: "hi", Streams: 1, Tokens: 60,
	})
	if err != nil {
		t.Fatal(err)
	}

	if res.DecodePasses != 30 {
		t.Fatalf("decode passes = %d, want 30 from the counter delta", res.DecodePasses)
	}
	if res.ServerTokens != 60 {
		t.Fatalf("server tokens = %d, want 60", res.ServerTokens)
	}
	if got := res.TokensPerPass; got < 1.99 || got > 2.01 {
		t.Fatalf("tokens per pass = %.2f, want 2.00", got)
	}
	if !res.MTPLoaded || !res.OnGPU {
		t.Errorf("placement not carried through: mtp=%v gpu=%v", res.MTPLoaded, res.OnGPU)
	}
}

// Warmup must not be counted, or it inflates the measured window.
func TestRemoteWarmupExcludedFromCounters(t *testing.T) {
	f := &fakeServer{tokensPerRequest: 20, passesPerRequest: 20}
	ts := httptest.NewServer(f.handler())
	defer ts.Close()

	r := NewRemote(ts.URL, "", "m", nil)
	res, err := RunRemote(context.Background(), r, Config{
		Model: "m", Prompt: "hi", Streams: 1, Tokens: 20, Warmup: 5,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Only the measured request should appear in the delta.
	if res.ServerTokens != 20 {
		t.Fatalf("server tokens = %d, want 20 — warmup must not be counted", res.ServerTokens)
	}
}

func TestRemoteCustomHeaderIsSent(t *testing.T) {
	// Deployments authenticate in different ways, so an arbitrary header must reach the
	// server rather than only a bearer token.
	f := &fakeServer{tokensPerRequest: 4, passesPerRequest: 4}
	ts := httptest.NewServer(f.handler())
	defer ts.Close()

	r := NewRemote(ts.URL, "", "m", map[string]string{"X-Test-Auth": "secret"})
	if _, err := r.Counters(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got, _ := f.authSeen.Load().(string); got != "secret" {
		t.Fatalf("custom header not sent; server saw %q", got)
	}
}

func TestRemoteRejectsUnusableConfig(t *testing.T) {
	r := NewRemote("http://127.0.0.1:1", "", "m", nil)
	for _, c := range []Config{
		{Streams: 0, Tokens: 10},
		{Streams: 1, Tokens: 1},
	} {
		if _, err := RunRemote(context.Background(), r, c); err == nil {
			t.Errorf("config %+v should be rejected", c)
		}
	}
}

func TestRemoteSurfacesServerErrors(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/v1/info") {
			_ = json.NewEncoder(w).Encode(map[string]any{"accelerated": false})
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer ts.Close()

	r := NewRemote(ts.URL, "", "m", nil)
	res, err := RunRemote(context.Background(), r, Config{Model: "m", Prompt: "x", Streams: 2, Tokens: 8})
	if err != nil {
		t.Fatal(err)
	}
	if res.Failures != 2 {
		t.Fatalf("failures = %d, want 2 — a failing endpoint must be reported, not averaged away",
			res.Failures)
	}
}
