// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package bench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Remote measures a deployed instance over HTTP.
//
// Client-side timing includes network, which local measurement does not, so the two are not
// interchangeable. What makes a remote run worth taking anyway is that the server's own
// counters can be read before and after: tokens per forward pass comes from the engine and is
// unaffected by network latency or a contended host, so it stays valid even when the
// throughput figures from the same run do not.
type Remote struct {
	BaseURL string
	APIKey  string
	Model   string
	Client  *http.Client

	// Headers are sent with every request. Some hosting providers authenticate with a
	// header of their own rather than the standard bearer scheme, and naming any one of
	// them here would put a provider's private detail in a public repository — so the
	// caller supplies whatever its deployment needs.
	Headers map[string]string
}

// NewRemote builds a client. The timeout is generous because a long generation on a busy host
// legitimately takes minutes.
func NewRemote(baseURL, apiKey, model string, headers map[string]string) *Remote {
	return &Remote{
		BaseURL: strings.TrimRight(baseURL, "/"),
		APIKey:  apiKey,
		Model:   model,
		Headers: headers,
		Client:  &http.Client{Timeout: 30 * time.Minute},
	}
}

func (r *Remote) do(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, r.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	if r.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.APIKey)
	}
	for k, v := range r.Headers {
		req.Header.Set(k, v)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return r.Client.Do(req)
}

// ServerCounters is the subset of the info endpoint a benchmark needs.
type ServerCounters struct {
	TokensGenerated uint64
	DecodePasses    uint64
	DraftsProposed  uint64
	DraftsAccepted  uint64
	RequestsTotal   uint64
	Evictions       uint64
	Accelerated     bool
	OnGPU           bool
	MTPLoaded       bool
	GPUFreeBytes    uint64
	Warning         string
}

// Counters reads the server's own view.
func (r *Remote) Counters(ctx context.Context) (ServerCounters, error) {
	var out ServerCounters
	resp, err := r.do(ctx, http.MethodGet, "/v1/info", nil)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()

	var info struct {
		Accelerated bool   `json:"accelerated"`
		Warning     string `json:"warning"`
		Devices     []struct {
			Type      string `json:"type"`
			FreeBytes uint64 `json:"free_bytes"`
		} `json:"devices"`
		Models []struct {
			Name  string `json:"name"`
			Stats *struct {
				TokensGenerated uint64 `json:"tokens_generated"`
				DecodePasses    uint64 `json:"decode_passes"`
				DraftsProposed  uint64 `json:"drafts_proposed"`
				DraftsAccepted  uint64 `json:"drafts_accepted"`
				RequestsTotal   uint64 `json:"requests_total"`
				EvictionsTotal  uint64 `json:"evictions_total"`
			} `json:"stats"`
			Placement *struct {
				OnGPU     bool `json:"on_gpu"`
				MTPLoaded bool `json:"mtp_loaded"`
			} `json:"placement"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return out, fmt.Errorf("info: %w", err)
	}

	out.Accelerated = info.Accelerated
	out.Warning = info.Warning
	for _, d := range info.Devices {
		if d.Type == "gpu" {
			out.GPUFreeBytes = d.FreeBytes
		}
	}
	for _, m := range info.Models {
		if r.Model != "" && m.Name != r.Model {
			continue
		}
		if m.Stats != nil {
			out.TokensGenerated = m.Stats.TokensGenerated
			out.DecodePasses = m.Stats.DecodePasses
			out.DraftsProposed = m.Stats.DraftsProposed
			out.DraftsAccepted = m.Stats.DraftsAccepted
			out.RequestsTotal = m.Stats.RequestsTotal
			out.Evictions = m.Stats.EvictionsTotal
		}
		if m.Placement != nil {
			out.OnGPU = m.Placement.OnGPU
			out.MTPLoaded = m.Placement.MTPLoaded
		}
		break
	}
	return out, nil
}

// stream issues one streaming completion and times it.
func (r *Remote) stream(ctx context.Context, prompt string, maxTokens int) (StreamResult, error) {
	body, _ := json.Marshal(map[string]any{
		"model":      r.Model,
		"stream":     true,
		"max_tokens": maxTokens,
		"messages":   []map[string]string{{"role": "user", "content": prompt}},
	})

	var res StreamResult
	submitted := time.Now()
	resp, err := r.do(ctx, http.MethodPost, "/v1/chat/completions", body)
	if err != nil {
		return res, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return res, fmt.Errorf("status %d", resp.StatusCode)
	}

	var first, last time.Time
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(payload), &chunk) != nil {
			continue
		}
		for _, c := range chunk.Choices {
			if c.Delta.Content == "" {
				continue
			}
			now := time.Now()
			if first.IsZero() {
				first = now
				res.TTFT = now.Sub(submitted)
			}
			last = now
			res.Tokens++
		}
	}
	res.Total = time.Since(submitted)
	if !first.IsZero() && last.After(first) {
		res.Decode = last.Sub(first)
	}
	return res, sc.Err()
}

// RunRemote measures a deployed instance, pairing client timings with the server's counters.
func RunRemote(ctx context.Context, r *Remote, cfg Config) (*Result, error) {
	if cfg.Streams < 1 || cfg.Tokens < 2 {
		return nil, fmt.Errorf("bench: need at least 1 stream and 2 tokens")
	}

	if cfg.Warmup > 0 {
		if _, err := r.stream(ctx, cfg.Prompt, cfg.Warmup); err != nil {
			return nil, fmt.Errorf("bench: warmup: %w", err)
		}
	}

	before, err := r.Counters(ctx)
	if err != nil {
		return nil, fmt.Errorf("bench: reading server counters: %w", err)
	}

	var (
		mu      sync.Mutex
		results = make([]StreamResult, cfg.Streams)
		firstAt time.Time
		lastAt  time.Time
	)

	start := time.Now()
	var wg sync.WaitGroup
	for i := 0; i < cfg.Streams; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sr, err := r.stream(ctx, cfg.Prompt, cfg.Tokens)
			sr.Index = i
			if err != nil {
				sr.Err = err.Error()
			}
			mu.Lock()
			results[i] = sr
			f := time.Now().Add(-sr.Total).Add(sr.TTFT)
			if sr.Decode > 0 {
				if firstAt.IsZero() || f.Before(firstAt) {
					firstAt = f
				}
				if e := f.Add(sr.Decode); e.After(lastAt) {
					lastAt = e
				}
			}
			mu.Unlock()
		}(i)
	}
	wg.Wait()
	wall := time.Since(start)

	after, err := r.Counters(ctx)
	if err != nil {
		return nil, fmt.Errorf("bench: reading server counters: %w", err)
	}

	res := &Result{Model: cfg.Model, Streams: cfg.Streams, Tokens: cfg.Tokens, Wall: wall}
	var ttfts []time.Duration
	for _, sr := range results {
		if sr.Err != "" {
			res.Failures++
			continue
		}
		res.TotalTokens += sr.Tokens
		ttfts = append(ttfts, sr.TTFT)
	}
	if wall > 0 {
		res.EndToEndTokPerSec = float64(res.TotalTokens) / wall.Seconds()
	}
	if !firstAt.IsZero() && lastAt.After(firstAt) {
		decoded := res.TotalTokens - (cfg.Streams - res.Failures)
		if decoded > 0 {
			res.DecodeTokPerSec = float64(decoded) / lastAt.Sub(firstAt).Seconds()
		}
	}
	if n := cfg.Streams - res.Failures; n > 0 {
		res.PerStreamTokPerSec = res.DecodeTokPerSec / float64(n)
	}

	// The server's own counters. These are the figures a contended host cannot distort,
	// because they are ratios of counts rather than rates against wall clock.
	res.DecodePasses = after.DecodePasses - before.DecodePasses
	// Differenced, not read absolute: the server may have served other traffic before this
	// run, and a lifetime ratio would report that traffic's acceptance rather than this
	// workload's.
	res.DraftsProposed = after.DraftsProposed - before.DraftsProposed
	res.DraftsAccepted = after.DraftsAccepted - before.DraftsAccepted
	if res.DraftsProposed > 0 {
		res.AcceptanceRate = float64(res.DraftsAccepted) / float64(res.DraftsProposed)
	}
	produced := after.TokensGenerated - before.TokensGenerated
	if res.DecodePasses > 0 {
		res.TokensPerPass = float64(produced) / float64(res.DecodePasses)
	}
	res.ServerTokens = produced
	res.Evictions = after.Evictions - before.Evictions
	res.MTPLoaded = after.MTPLoaded
	res.OnGPU = after.OnGPU

	res.TTFTp50 = percentile(ttfts, 0.50)
	res.TTFTp95 = percentile(ttfts, 0.95)
	res.TTFTMax = percentile(ttfts, 1.0)
	return res, nil
}

// VerifySpeculation asks the same prompt twice — once with drafting and once without — and
// reports whether the answers match.
//
// This is the check that speculation is an optimisation and nothing more. It is worth having
// as a command because no counter shows it: acceptance, tokens-per-pass and a clean error log
// all look healthy when the caches disagree and the text quietly degrades.
//
// It runs the two requests sequentially at one stream on purpose. Continuous batching
// interleaves concurrent requests differently on every run, and that alone changes batch
// composition enough to flip a near-tied token — so a difference measured under concurrency
// says nothing about speculation. Sequential single-stream is the only arrangement where the
// comparison has a stable control.
func (r *Remote) VerifySpeculation(ctx context.Context, prompt string, maxTokens int) (VerifyResult, error) {
	var out VerifyResult
	off := false

	withSpec, err := r.completeOnce(ctx, prompt, maxTokens, nil)
	if err != nil {
		return out, fmt.Errorf("with speculation: %w", err)
	}
	withoutSpec, err := r.completeOnce(ctx, prompt, maxTokens, &off)
	if err != nil {
		return out, fmt.Errorf("without speculation: %w", err)
	}

	out.WithSpeculation = withSpec
	out.WithoutSpeculation = withoutSpec
	out.Identical = withSpec == withoutSpec
	if !out.Identical {
		out.FirstDifference = -1
		for i := 0; i < len(withSpec) && i < len(withoutSpec); i++ {
			if withSpec[i] != withoutSpec[i] {
				out.FirstDifference = i
				break
			}
		}
		if out.FirstDifference < 0 {
			// One is a prefix of the other, so they part where the shorter ends.
			out.FirstDifference = min(len(withSpec), len(withoutSpec))
		}
	}
	return out, nil
}

// VerifyResult is the outcome of a speculation-neutrality check.
type VerifyResult struct {
	WithSpeculation    string `json:"with_speculation"`
	WithoutSpeculation string `json:"without_speculation"`
	Identical          bool   `json:"identical"`
	// FirstDifference is the byte offset where the answers part, or 0 when identical.
	FirstDifference int `json:"first_difference"`
}

func (r *Remote) completeOnce(ctx context.Context, prompt string, maxTokens int, speculate *bool) (string, error) {
	body := map[string]any{
		"model":    r.Model,
		"messages": []map[string]string{{"role": "user", "content": prompt}},
		// Greedy, so the only thing that can move the answer is the engine.
		"temperature": 0,
		"max_tokens":  maxTokens,
	}
	if speculate != nil {
		body["speculate"] = *speculate
	}
	buf, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	resp, err := r.do(ctx, http.MethodPost, "/v1/chat/completions", buf)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("status %d", resp.StatusCode)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("no choices returned")
	}
	return out.Choices[0].Message.Content, nil
}
