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

// HasChatTemplate reports whether this model can render chat messages.
func (r *Runner) HasChatTemplate() bool { return r.chatTmpl != "" }
