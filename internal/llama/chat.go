// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

/*
#include <stdlib.h>
#include "llama.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/sideblank/llama-herd/internal/engine"
)

// ChatMessage is one turn in a conversation.
type ChatMessage struct {
	Role    string
	Content string
}

// ErrNoChatTemplate means the model file carries no chat template, so messages cannot be
// rendered the way it was trained to expect.
var ErrNoChatTemplate = errors.New("llama: model has no chat template")

// ChatTemplate returns the template embedded in the model file, or ErrNoChatTemplate.
//
// Using the model's own template matters more than it looks: a model rendered with the wrong
// role markers still produces fluent text, so the failure shows up as quietly worse output
// rather than an error.
func (m *Model) ChatTemplate() (string, error) {
	p := C.llama_model_chat_template(m.c, nil)
	if p == nil {
		return "", ErrNoChatTemplate
	}
	return C.GoString(p), nil
}

// ApplyChatTemplate renders messages into a prompt. An empty tmpl uses the model's own
// template. addAssistant appends the tokens that open an assistant turn, which is what makes
// the model continue as the assistant rather than predicting more conversation.
func ApplyChatTemplate(tmpl string, msgs []ChatMessage, addAssistant bool) (string, error) {
	if len(msgs) == 0 {
		return "", errors.New("llama: no messages to render")
	}

	cmsgs := make([]C.struct_llama_chat_message, len(msgs))
	for i, m := range msgs {
		role := C.CString(m.Role)
		content := C.CString(m.Content)
		defer C.free(unsafe.Pointer(role))
		defer C.free(unsafe.Pointer(content))
		cmsgs[i].role = role
		cmsgs[i].content = content
	}

	var ctmpl *C.char
	if tmpl != "" {
		ctmpl = C.CString(tmpl)
		defer C.free(unsafe.Pointer(ctmpl))
	}

	// The documented sizing hint is twice the total message length; a short reply from
	// the template can still exceed it, so a too-small buffer is retried at the exact
	// size the call reports rather than guessed at again.
	size := 0
	for _, m := range msgs {
		size += len(m.Role) + len(m.Content)
	}
	size = size*2 + 512

	buf := make([]byte, size)
	n := C.llama_chat_apply_template(ctmpl, &cmsgs[0], C.size_t(len(cmsgs)),
		C.bool(addAssistant), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)))
	if n < 0 {
		return "", fmt.Errorf("llama: could not apply chat template (%d)", int32(n))
	}
	if int(n) > len(buf) {
		buf = make([]byte, int(n))
		n = C.llama_chat_apply_template(ctmpl, &cmsgs[0], C.size_t(len(cmsgs)),
			C.bool(addAssistant), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)))
		if n < 0 || int(n) > len(buf) {
			return "", fmt.Errorf("llama: chat template did not fit (%d)", int32(n))
		}
	}
	return string(buf[:n]), nil
}

// RenderChat renders messages with this model's own template.
//
// Safe for concurrent use: the template was captured at load time, so this touches no
// context or sampler state and does not contend with the decode loop.
func (r *Runner) RenderChat(msgs []engine.ChatMessage) (string, error) {
	if r.chatTmpl == "" {
		return "", ErrNoChatTemplate
	}
	conv := make([]ChatMessage, len(msgs))
	for i, m := range msgs {
		conv[i] = ChatMessage{Role: m.Role, Content: m.Content}
	}
	return ApplyChatTemplate(r.chatTmpl, conv, true)
}

// RenderChatTools renders messages together with tool definitions, using this model's own
// template.
//
// An empty toolsJSON renders exactly as RenderChat does, so callers can pass whatever the
// request carried without branching. Tools reach the model only through this path: the core
// template call takes messages alone and drops a tools array without reporting it.
func (r *Runner) RenderChatTools(msgs []engine.ChatMessage, toolsJSON, toolChoice string, think bool) (string, error) {
	if toolsJSON == "" {
		return r.RenderChatThinking(msgs, think)
	}
	conv := make([]ChatMessage, len(msgs))
	for i, m := range msgs {
		conv[i] = ChatMessage{Role: m.Role, Content: m.Content}
	}
	// ⛔ think goes INTO the template, and the prime is NOT appended afterwards. This path
	// renders the model's real Jinja template, which branches on enable_thinking and emits the
	// CLOSED block itself when thinking is off. Appending the prime on top of that nests a
	// second reasoning block inside the first, unclosed — which is what this code did before,
	// because the shim defaulted enable_thinking to true and the prime went on regardless.
	// The core-template path still needs the prime: it knows nothing about thinking at all.
	return r.model.ApplyChatTemplateTools(conv, toolsJSON, toolChoice, true, think)
}

// ParseChatOutput separates a completion into prose and the tool calls the model asked for.
//
// The arguments must match the render — think included: the parser comes from the same template
// application and carries its think tags, so one built from different inputs reads the text with
// the wrong grammar and finds nothing. It reports that as zero calls, not as an error, which is
// why the caller must not treat "no calls" as proof the model declined.
func (r *Runner) ParseChatOutput(msgs []engine.ChatMessage, toolsJSON, toolChoice, text string, think bool) (string, []engine.ToolCall, error) {
	conv := make([]ChatMessage, len(msgs))
	for i, m := range msgs {
		conv[i] = ChatMessage{Role: m.Role, Content: m.Content}
	}
	out, err := r.model.ParseOutput(conv, toolsJSON, toolChoice, text, think)
	if err != nil {
		return text, nil, err
	}
	calls := make([]engine.ToolCall, 0, len(out.ToolCalls))
	for _, c := range out.ToolCalls {
		calls = append(calls, engine.ToolCall{Name: c.Name, Arguments: c.Arguments, ID: c.ID})
	}
	return out.Content, calls, nil
}

// SupportsTools reports whether this model can be given tool definitions.
//
// It answers from the template rather than from a build flag: a model with no chat template
// cannot be told about tools at all, and saying otherwise would let a request be accepted and
// then answered without them.
func (r *Runner) SupportsTools() bool { return r.chatTmpl != "" }

// DefaultNoThinkPrime opens and immediately closes a reasoning block, so a model that would
// otherwise reason continues straight into its answer.
//
// The blank line matters: these templates are trained with the block followed by a blank
// line before the answer, and priming without it leaves the model completing into a shape it
// has not seen.
const DefaultNoThinkPrime = "<think>\n\n</think>\n\n"

// SupportsThinking reports whether this model has a reasoning block to suppress.
//
// Detected rather than configured: a model that tokenizes "<think>" to exactly one token has
// it as a real special token, which is what the reasoning templates use. A model that
// tokenizes it to several has only the literal characters, and priming would put stray text
// into its prompt.
func (r *Runner) SupportsThinking() bool { return r.thinkPrime != "" }

// RenderChatThinking renders a chat, suppressing the reasoning block unless think is true.
func (r *Runner) RenderChatThinking(msgs []engine.ChatMessage, think bool) (string, error) {
	out, err := r.RenderChat(msgs)
	if err != nil {
		return "", err
	}
	if think || r.thinkPrime == "" {
		return out, nil
	}
	return out + r.thinkPrime, nil
}

// HasChatTemplate reports whether this model can render chat messages.
func (r *Runner) HasChatTemplate() bool { return r.chatTmpl != "" }
