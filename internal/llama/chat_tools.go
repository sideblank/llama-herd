// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

/*
#cgo LDFLAGS: -llhchat
#include <stdlib.h>
#include "lhchat.h"
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"strings"
	"unsafe"
)

// ToolCall is one call a model asked for, in the shape the OpenAI API returns.
type ToolCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	ID        string `json:"id"`
}

// ParsedOutput is a completion separated into what the model said and what it asked to run.
type ParsedOutput struct {
	Content          string     `json:"content"`
	ReasoningContent string     `json:"reasoning_content"`
	ToolCalls        []ToolCall `json:"tool_calls"`
}

// encodeMessages is the shim's wire shape: "role\x1fcontent" records joined by \x1e.
func encodeMessages(msgs []ChatMessage) string {
	var sb strings.Builder
	for i, m := range msgs {
		if i > 0 {
			sb.WriteByte(0x1e)
		}
		sb.WriteString(m.Role)
		sb.WriteByte(0x1f)
		sb.WriteString(m.Content)
	}
	return sb.String()
}

// toolChoiceCode maps the API's string onto the shim's integer.
func toolChoiceCode(choice string) C.int32_t {
	switch choice {
	case "required":
		return 1
	case "none":
		return 2
	default:
		return 0
	}
}

// callWithGrowingBuffer runs a shim call that returns bytes written, or the negative of the
// size it needs. A negative small enough to be an error code is reported as one rather than
// retried, so a refusal is not mistaken for a buffer that was merely too small.
func callWithGrowingBuffer(first int, call func(buf *C.char, size C.int32_t) C.int32_t) (string, error) {
	size := first
	buf := C.malloc(C.size_t(size))
	defer C.free(buf)
	n := call((*C.char)(buf), C.int32_t(size))
	if n < 0 {
		if -n <= 3 {
			return "", fmt.Errorf("llama: chat shim refused (%d)", int(n))
		}
		size = int(-n)
		C.free(buf)
		buf = C.malloc(C.size_t(size))
		n = call((*C.char)(buf), C.int32_t(size))
		if n < 0 {
			return "", fmt.Errorf("llama: chat shim refused after resize (%d)", int(n))
		}
	}
	return C.GoStringN((*C.char)(buf), C.int(n)), nil
}

// ApplyChatTemplateTools renders messages and tool definitions into a prompt using the model's
// own chat template.
//
// toolsJSON is an OpenAI `tools` array verbatim; empty renders without tools, so this is a
// superset of ApplyChatTemplate rather than an alternative to it. Because the model's template
// does the formatting, a model with an unusual tool syntax is handled correctly without this
// package knowing anything about it.
// think selects the template's reasoning branch. Passing false makes the template emit the
// CLOSED block itself; a caller that does that must NOT also append a no-think prime, or the two
// nest. The core-template path needs the prime because it knows nothing about thinking — this
// one does not.
func (m *Model) ApplyChatTemplateTools(msgs []ChatMessage, toolsJSON, toolChoice string, addAssistant, think bool) (string, error) {
	if len(msgs) == 0 {
		return "", ErrNoChatTemplate
	}
	blob := C.CString(encodeMessages(msgs))
	defer C.free(unsafe.Pointer(blob))
	tools := C.CString(toolsJSON)
	defer C.free(unsafe.Pointer(tools))

	addAss := C.int(0)
	if addAssistant {
		addAss = 1
	}
	thinking := C.int(0)
	if think {
		thinking = 1
	}
	hint := len(toolsJSON) + 2048
	for _, msg := range msgs {
		hint += (len(msg.Role) + len(msg.Content)) * 2
	}
	return callWithGrowingBuffer(hint, func(buf *C.char, size C.int32_t) C.int32_t {
		return C.lhchat_apply_template_tools_ex(unsafe.Pointer(m.c), blob, tools,
			toolChoiceCode(toolChoice), addAss, thinking, buf, size)
	})
}

// ParseOutput separates a completion into prose and the tool calls the model asked for.
//
// The messages, tools and choice must be the ones the prompt was rendered with. The parser is
// derived from that render, and one derived from different inputs reads the same text with the
// wrong grammar — reporting no calls rather than an error.
// think must be the SAME value the render used. The parser configuration is derived from a
// re-render, and it carries the think tags; rendering with thinking off and parsing with it on
// reads the text with the wrong grammar and reports no calls rather than an error.
func (m *Model) ParseOutput(msgs []ChatMessage, toolsJSON, toolChoice, text string, think bool) (ParsedOutput, error) {
	var out ParsedOutput
	blob := C.CString(encodeMessages(msgs))
	defer C.free(unsafe.Pointer(blob))
	tools := C.CString(toolsJSON)
	defer C.free(unsafe.Pointer(tools))
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	thinking := C.int(0)
	if think {
		thinking = 1
	}

	raw, err := callWithGrowingBuffer(len(text)*2+1024, func(buf *C.char, size C.int32_t) C.int32_t {
		return C.lhchat_parse_output_ex(unsafe.Pointer(m.c), blob, tools,
			toolChoiceCode(toolChoice), ctext, thinking, buf, size)
	})
	if err != nil {
		return out, err
	}
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return out, fmt.Errorf("llama: parsing tool calls: %w", err)
	}
	return out, nil
}
