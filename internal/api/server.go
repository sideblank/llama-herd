// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/sideblank/llama-herd/internal/engine"
)

// Clock is injectable so responses are deterministic under test.
type Clock func() time.Time

// Server exposes a registry over HTTP.
type Server struct {
	reg   *engine.Registry
	now   Clock
	seq   atomic.Uint64
	idPfx string
}

// New builds a server over reg.
func New(reg *engine.Registry) *Server {
	return &Server{reg: reg, now: time.Now, idPfx: "chatcmpl-"}
}

// Handler returns the routed HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /v1/models", s.listModels)
	mux.HandleFunc("POST /v1/chat/completions", s.chatCompletions)
	return mux
}

func (s *Server) writeErr(w http.ResponseWriter, status int, kind, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(apiError{apiErrorBody{Message: msg, Type: kind}})
}

// health reports per-model liveness. A model whose decode loop has died makes this
// unhealthy even though the process is still listening — otherwise a load balancer keeps
// sending traffic to a server that cannot answer.
func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	type modelHealth struct {
		Name  string `json:"name"`
		OK    bool   `json:"ok"`
		Error string `json:"error,omitempty"`
	}
	var out struct {
		OK     bool          `json:"ok"`
		Models []modelHealth `json:"models"`
	}
	out.OK = true
	for _, name := range s.reg.Names() {
		err := s.reg.Health()[name]
		mh := modelHealth{Name: name, OK: err == nil}
		if err != nil {
			mh.Error = err.Error()
			out.OK = false
		}
		out.Models = append(out.Models, mh)
	}

	w.Header().Set("Content-Type", "application/json")
	if !out.OK {
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	_ = json.NewEncoder(w).Encode(out)
}

func (s *Server) listModels(w http.ResponseWriter, _ *http.Request) {
	list := modelList{Object: "list"}
	created := s.now().Unix()
	for _, name := range s.reg.Names() {
		list.Data = append(list.Data, modelInfo{
			ID: name, Object: "model", Created: created, OwnedBy: "llama-herd",
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(list)
}

func (s *Server) nextID() string {
	return fmt.Sprintf("%s%d", s.idPfx, s.seq.Add(1))
}

func (s *Server) chatCompletions(w http.ResponseWriter, r *http.Request) {
	// Decode leniently on purpose. Real clients send fields this server does not
	// implement — temperature, tools, penalties — and rejecting a request because it
	// carried one would break callers over something they cannot control. Unsupported
	// fields are ignored rather than honoured, which is visible in the response.
	var req ChatRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxRequestBytes))
	if err := dec.Decode(&req); err != nil {
		s.writeErr(w, http.StatusBadRequest, "invalid_request_error",
			fmt.Sprintf("could not parse request: %v", err))
		return
	}

	if req.Model == "" {
		s.writeErr(w, http.StatusBadRequest, "invalid_request_error", "model is required")
		return
	}
	if len(req.Messages) == 0 {
		s.writeErr(w, http.StatusBadRequest, "invalid_request_error", "messages must not be empty")
		return
	}

	eng, err := s.reg.Get(req.Model)
	if err != nil {
		switch {
		case errors.Is(err, engine.ErrUnknownModel):
			s.writeErr(w, http.StatusNotFound, "invalid_request_error", err.Error())
		case errors.Is(err, engine.ErrModelDead):
			s.writeErr(w, http.StatusServiceUnavailable, "server_error", err.Error())
		default:
			s.writeErr(w, http.StatusInternalServerError, "server_error", err.Error())
		}
		return
	}

	rend, err := s.reg.Renderer(req.Model)
	if err != nil {
		s.writeErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	prompt, err := rend.RenderChat(req.Messages)
	if err != nil {
		s.writeErr(w, http.StatusBadRequest, "invalid_request_error",
			fmt.Sprintf("could not render messages: %v", err))
		return
	}

	stream, err := eng.Submit(r.Context(), engine.Request{
		Prompt:    prompt,
		MaxTokens: req.limit(),
		Stop:      req.Stop,
		Sampling:  req.sampling(),
	})
	if err != nil {
		status := http.StatusBadRequest
		kind := "invalid_request_error"
		if errors.Is(err, engine.ErrQueueFull) {
			// Backpressure, not a client mistake: say so with the status that tells
			// a caller to retry rather than to change the request.
			status, kind = http.StatusTooManyRequests, "rate_limit_error"
		}
		s.writeErr(w, status, kind, err.Error())
		return
	}
	defer stream.Close()

	if req.Stream {
		s.streamChat(w, req.Model, stream)
		return
	}
	s.bufferChat(w, req.Model, stream)
}

// maxRequestBytes bounds a request body. Prompts can be large — 128k of context is the
// point — so this is generous, but unbounded would let one caller exhaust memory.
const maxRequestBytes = 64 << 20

func (s *Server) bufferChat(w http.ResponseWriter, model string, st *engine.Stream) {
	var sb strings.Builder
	reason := engine.ReasonEOS
	for ev := range st.Events {
		if ev.Err != nil {
			s.writeErr(w, http.StatusInternalServerError, "server_error", ev.Err.Error())
			return
		}
		sb.WriteString(ev.Text)
		if ev.Done {
			reason = ev.Reason
		}
	}

	fr := finishReason(reason)
	resp := chatResponse{
		ID: s.nextID(), Object: "chat.completion", Created: s.now().Unix(), Model: model,
		Choices: []choice{{
			Index:        0,
			Message:      &respMessage{Role: "assistant", Content: sb.String()},
			FinishReason: &fr,
		}},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// streamChat writes server-sent events in the shape clients expect: a role-only first chunk,
// content deltas, a final chunk carrying finish_reason, then [DONE].
func (s *Server) streamChat(w http.ResponseWriter, model string, st *engine.Stream) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		s.writeErr(w, http.StatusInternalServerError, "server_error", "streaming unsupported")
		return
	}

	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	// Without this, a reverse proxy may buffer the whole response and defeat streaming.
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	id := s.nextID()
	created := s.now().Unix()

	send := func(c choice) {
		chunk := chatResponse{
			ID: id, Object: "chat.completion.chunk", Created: created,
			Model: model, Choices: []choice{c},
		}
		b, err := json.Marshal(chunk)
		if err != nil {
			return
		}
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}

	send(choice{Index: 0, Delta: &respMessage{Role: "assistant"}})

	reason := engine.ReasonEOS
	for ev := range st.Events {
		if ev.Err != nil {
			// The status line is long gone, so the error has to ride the stream.
			send(choice{Index: 0, Delta: &respMessage{Content: ""}})
			fmt.Fprintf(w, "data: %s\n\n", mustJSON(apiError{apiErrorBody{
				Message: ev.Err.Error(), Type: "server_error",
			}}))
			flusher.Flush()
			break
		}
		if ev.Text != "" {
			send(choice{Index: 0, Delta: &respMessage{Content: ev.Text}})
		}
		if ev.Done {
			reason = ev.Reason
		}
	}

	fr := finishReason(reason)
	send(choice{Index: 0, Delta: &respMessage{}, FinishReason: &fr})
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func mustJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}

// finishReason maps the engine's stop reasons onto the vocabulary clients understand.
func finishReason(r string) string {
	switch r {
	case engine.ReasonLength, engine.ReasonContext:
		return "length"
	case engine.ReasonStopSeq, engine.ReasonEOS:
		return "stop"
	case engine.ReasonCancel:
		return "stop"
	default:
		return "stop"
	}
}
