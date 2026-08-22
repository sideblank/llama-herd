// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package api serves an OpenAI-compatible HTTP surface.
//
// Compatibility is the point: any agent or editor plugin with a configurable base URL works
// against this without a plugin, a fork, or a per-tool integration.
package api

import "github.com/sideblank/llama-herd/internal/engine"

// ChatRequest is the subset of the chat-completions request this server honours.
type ChatRequest struct {
	Model string `json:"model"`
	// Messages keep their raw content so both the string and array-of-parts shapes
	// survive decoding; they are reduced to text plus media once the model's marker is
	// known, since the marker is model-specific.
	Messages []message `json:"messages"`
	Stream   bool      `json:"stream"`

	MaxTokens int `json:"max_tokens,omitempty"`
	// MaxCompletionTokens is the newer spelling. When both are present it wins.
	MaxCompletionTokens int `json:"max_completion_tokens,omitempty"`

	// Stop accepts either a string or an array, which is why it is decoded loosely.
	Stop stopValue `json:"stop,omitempty"`

	// Think asks the model to reason before answering. Reasoning is off by default: left to
	// itself a reasoning model decides per request, and the decision is often a near-tie, so
	// the same prompt costs an empty block on one run and hundreds of tokens on the next.
	// Those tokens are generated and paid for without being answer text.
	//
	// It is a request-level choice rather than a deployment-level one because the caller is
	// the one who knows whether the question is worth reasoning about.
	Think *bool `json:"think,omitempty"`

	// Speculate turns drafting off for this request when explicitly false. Omitting it
	// follows the model's configuration.
	//
	// This exists to make one check possible against a running server: the same prompt,
	// answered with speculation and without, must produce the same text. Speculation is an
	// optimisation, so any difference is a defect — and no counter reveals it, since
	// acceptance and tokens-per-pass look healthy either way.
	Speculate *bool `json:"speculate,omitempty"`

	// Sampling controls. Pointers so an explicit zero is distinguishable from absence:
	// "temperature": 0 asks for greedy decoding, which is not the same as omitting it.
	Temperature      *float32 `json:"temperature,omitempty"`
	TopP             *float32 `json:"top_p,omitempty"`
	TopK             *int32   `json:"top_k,omitempty"`
	MinP             *float32 `json:"min_p,omitempty"`
	FrequencyPenalty *float32 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float32 `json:"presence_penalty,omitempty"`
	RepeatPenalty    *float32 `json:"repetition_penalty,omitempty"`
	Seed             *uint32  `json:"seed,omitempty"`
}

// sampling maps the request onto the engine's neutral representation. Fields the caller
// omitted stay nil, so the model's configured defaults survive.
func (r ChatRequest) sampling() *engine.SamplingParams {
	p := &engine.SamplingParams{
		Temperature:     r.Temperature,
		TopP:            r.TopP,
		TopK:            r.TopK,
		MinP:            r.MinP,
		FreqPenalty:     r.FrequencyPenalty,
		PresencePenalty: r.PresencePenalty,
		RepeatPenalty:   r.RepeatPenalty,
		Seed:            r.Seed,
	}
	if p.IsZero() {
		return nil
	}
	return p
}

// limit resolves the two spellings of the output cap.
func (r ChatRequest) limit() int {
	if r.MaxCompletionTokens > 0 {
		return r.MaxCompletionTokens
	}
	return r.MaxTokens
}

// Message shapes for responses.
type respMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type choice struct {
	Index        int          `json:"index"`
	Message      *respMessage `json:"message,omitempty"`
	Delta        *respMessage `json:"delta,omitempty"`
	FinishReason *string      `json:"finish_reason"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type chatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []choice `json:"choices"`
	Usage   *usage   `json:"usage,omitempty"`
}

// modelInfo is one entry of GET /v1/models.
type modelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type modelList struct {
	Object string      `json:"object"`
	Data   []modelInfo `json:"data"`
}

// apiError matches the error envelope clients expect, so a failure surfaces as a message
// rather than an opaque status code.
type apiError struct {
	Error apiErrorBody `json:"error"`
}

type apiErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}
