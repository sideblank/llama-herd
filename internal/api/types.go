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
	Model    string               `json:"model"`
	Messages []engine.ChatMessage `json:"messages"`
	Stream   bool                 `json:"stream"`

	MaxTokens int `json:"max_tokens,omitempty"`
	// MaxCompletionTokens is the newer spelling. When both are present it wins.
	MaxCompletionTokens int `json:"max_completion_tokens,omitempty"`

	// Stop accepts either a string or an array, which is why it is decoded loosely.
	Stop stopValue `json:"stop,omitempty"`
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
