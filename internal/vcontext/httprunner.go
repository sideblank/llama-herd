// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// HTTPRunner executes requests against a llama-herd over its chat-completions API.
//
// Over the public API rather than the engine directly, deliberately: this layer is a client of the
// engine, so it works against any llama-herd — local, remote, or several — and nothing here can
// reach past the contract other callers use.
type HTTPRunner struct {
	// BaseURL is the engine root, e.g. http://localhost:8080
	BaseURL string
	// Model is the manifest name to address.
	Model string
	// Client is optional; a sensible one is used when nil.
	Client *http.Client
}

// NewHTTPRunner returns a runner with a timeout suited to chunk work.
//
// The timeout is generous because a chunk carries thousands of tokens of prefill: on a card
// measured at ~3,300 tok/s an 8k chunk is seconds, and a queued one waits behind the herd. Too
// tight a deadline turns ordinary queueing into a failed chunk, which fails the whole batch.
func NewHTTPRunner(baseURL, model string) *HTTPRunner {
	return &HTTPRunner{
		BaseURL: baseURL,
		Model:   model,
		Client:  &http.Client{Timeout: 10 * time.Minute},
	}
}

type chatReq struct {
	Model       string    `json:"model"`
	Messages    []chatMsg `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float32   `json:"temperature"`
	Grammar     string    `json:"grammar,omitempty"`
}

type chatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResp struct {
	Choices []struct {
		Message chatMsg `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func (h *HTTPRunner) Run(ctx context.Context, req Request) (string, error) {
	body, err := json.Marshal(chatReq{
		Model:    h.Model,
		Messages: []chatMsg{{Role: "user", Content: req.Prompt}},
		// Chunk work is extraction, not composition: the same chunk must give the same digest
		// every time or a merge is not reproducible and a conflict cannot be trusted.
		Temperature: 0,
		MaxTokens:   req.MaxTokens,
		Grammar:     req.Grammar,
	})
	if err != nil {
		return "", err
	}

	hr, err := http.NewRequestWithContext(ctx, http.MethodPost,
		h.BaseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	hr.Header.Set("Content-Type", "application/json")

	c := h.Client
	if c == nil {
		c = http.DefaultClient
	}
	resp, err := c.Do(hr)
	if err != nil {
		return "", fmt.Errorf("vcontext: chunk %d: %w", req.Chunk, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("vcontext: chunk %d: reading response: %w", req.Chunk, err)
	}

	var out chatResp
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", fmt.Errorf("vcontext: chunk %d: %s returned unparseable JSON: %w",
			req.Chunk, resp.Status, err)
	}
	if out.Error != nil {
		return "", fmt.Errorf("vcontext: chunk %d: %s", req.Chunk, out.Error.Message)
	}
	// A present-but-empty choices array is what a truncated or filtered response looks like, and
	// indexing it directly would panic where a clear error belongs.
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("vcontext: chunk %d: %s returned no choices", req.Chunk, resp.Status)
	}
	return out.Choices[0].Message.Content, nil
}
