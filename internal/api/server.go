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
	"github.com/sideblank/llama-herd/internal/hostinfo"
)

// Clock is injectable so responses are deterministic under test.
type Clock func() time.Time

// Server exposes a registry over HTTP.
type Server struct {
	reg        *engine.Registry
	now        Clock
	seq        atomic.Uint64
	idPfx      string
	devices    Devices
	build      BuildInfo
	placements map[string]func() Placement
}

// BuildInfo describes the running binary.
type BuildInfo struct {
	Version     string `json:"version"`
	Commit      string `json:"commit"`
	LlamaCppRef string `json:"llama_cpp_ref"`
}

// WithDevices makes the server report the hardware it found.
func (s *Server) WithDevices(d Devices) *Server { s.devices = d; return s }

// WithBuild makes the server report which build is running.
func (s *Server) WithBuild(b BuildInfo) *Server { s.build = b; return s }

// WithPlacement records how one model was loaded, so the server can report whether the
// weights actually reached an accelerator.
func (s *Server) WithPlacement(model string, f func() Placement) *Server {
	if s.placements == nil {
		s.placements = map[string]func() Placement{}
	}
	s.placements[model] = f
	return s
}

// New builds a server over reg.
func New(reg *engine.Registry) *Server {
	return &Server{reg: reg, now: time.Now, idPfx: "chatcmpl-"}
}

// Handler returns the routed HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /v1/info", s.info)
	mux.HandleFunc("GET /metrics", s.metrics)
	mux.HandleFunc("GET /v1/models", s.listModels)
	mux.HandleFunc("POST /v1/chat/completions", s.chatCompletions)
	return mux
}

func (s *Server) writeErr(w http.ResponseWriter, status int, kind, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(apiError{apiErrorBody{Message: msg, Type: kind}})
}

// Devices is set by the server owner to report what hardware the process actually found.
//
// This exists because the most damaging failure in a GPU runtime is silent: when backends
// are not registered or the driver is absent, the server finds no accelerator, runs on CPU,
// and reports itself perfectly healthy. Nothing in a chat response reveals it — only the
// throughput, and only if you know what to expect. A deployed instance must be able to say
// what it is running on.
type Devices func() []DeviceInfo

// DeviceInfo is one accelerator or CPU the process can see.
type DeviceInfo struct {
	Index       int    `json:"index"`
	Name        string `json:"name"`
	Type        string `json:"type"`
	TotalBytes  uint64 `json:"total_bytes"`
	FreeBytes   uint64 `json:"free_bytes"`
	Description string `json:"description,omitempty"`
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

// Info is the full picture of a running instance: what it is, what it runs on, and what it
// is currently doing. Intended to be safe to expose to whoever operates the process,
// including an end user running it on their own machine.
type Info struct {
	Build       BuildInfo     `json:"build"`
	Accelerated bool          `json:"accelerated"`
	Warning     string        `json:"warning,omitempty"`
	Host        hostinfo.Host `json:"host"`
	Devices     []DeviceInfo  `json:"devices"`
	Models      []ModelStatus `json:"models"`
}

// ModelStatus is one model's configuration and live utilisation.
type ModelStatus struct {
	Name  string        `json:"name"`
	OK    bool          `json:"ok"`
	Error string        `json:"error,omitempty"`
	Stats *engine.Stats `json:"stats,omitempty"`

	// Placement records where this model's weights actually went.
	//
	// A device being present is not the same as the model running on it. Layers can fail
	// to offload, or be configured not to, and the server then reports an accelerator
	// while doing the work on CPU — which looks like a mysteriously slow GPU rather than
	// a model that never reached it.
	Placement *Placement `json:"placement,omitempty"`
}

// Placement describes how a model was loaded.
type Placement struct {
	// GPULayersRequested is what the manifest asked for; -1 means all.
	GPULayersRequested int32 `json:"gpu_layers_requested"`
	// LayersTotal is the model's layer count.
	LayersTotal int32 `json:"layers_total"`
	// OnGPU is false when the weights are on CPU regardless of what devices exist.
	OnGPU bool `json:"on_gpu"`

	ContextTotal  uint32 `json:"context_total"`
	ContextPerSeq uint32 `json:"context_per_stream"`
	BatchSize     uint32 `json:"batch_size"`
	KVTypeK       string `json:"kv_type_k"`
	KVTypeV       string `json:"kv_type_v"`
	FlashAttn     bool   `json:"flash_attention"`
	MTPLoaded     bool   `json:"mtp_loaded"`
}

// PlacementSource is implemented by a backend that can describe where its weights went.
type PlacementSource interface {
	Placement() Placement
}

// snapshot builds the current Info.
func (s *Server) snapshot() Info {
	var in Info
	in.Build = s.build
	in.Host = hostinfo.Read()

	if s.devices != nil {
		in.Devices = s.devices()
	}
	for _, d := range in.Devices {
		if d.Type == "gpu" {
			in.Accelerated = true
		}
	}
	health := s.reg.Health()
	for _, name := range s.reg.Names() {
		ms := ModelStatus{Name: name, OK: health[name] == nil}
		if err := health[name]; err != nil {
			ms.Error = err.Error()
		}
		if eng, err := s.reg.Get(name); err == nil {
			st := eng.Stats()
			ms.Stats = &st
		}
		if p, ok := s.placements[name]; ok {
			pp := p()
			ms.Placement = &pp
		}
		in.Models = append(in.Models, ms)
	}

	// Computed last, because the placement check needs the model list. Ordered by
	// severity: no accelerator at all, then an accelerator the weights never reached,
	// then a machine too busy to use what it has.
	switch offloaded, total := placementSummary(in.Models); {
	case !in.Accelerated:
		in.Warning = "no dedicated-memory GPU found — this process is running on CPU. " +
			"Requests will be answered correctly but far more slowly."
	case total > 0 && offloaded < total:
		in.Warning = fmt.Sprintf(
			"a GPU is present but %d of %d model(s) are running on CPU — the accelerator was "+
				"found and the weights did not reach it, which reads as a slow GPU rather than "+
				"an unused one.", total-offloaded, total)
	case in.Host.Oversubscribed():
		in.Warning = "the machine is loaded beyond its core count, so throughput will be " +
			"well below what this hardware can do, for reasons unrelated to the model."
	}

	return in
}

// placementSummary counts how many models actually reached an accelerator.
func placementSummary(ms []ModelStatus) (onGPU, total int) {
	for _, m := range ms {
		if m.Placement == nil {
			continue
		}
		total++
		if m.Placement.OnGPU {
			onGPU++
		}
	}
	return onGPU, total
}

// info reports the build and the hardware in use.
//
// The accelerated field is the one that matters: a server that found no accelerator still
// answers requests correctly, just far more slowly, so this is the only cheap way to tell a
// working deployment from a silently degraded one.
func (s *Server) info(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(s.snapshot())
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

	marker, acceptsMedia := eng.MediaMarker()
	msgs := make([]engine.ChatMessage, 0, len(req.Messages))
	var media [][]byte
	for i, m := range req.Messages {
		text, mm, err := m.parse(marker)
		if err != nil {
			s.writeErr(w, http.StatusBadRequest, "invalid_request_error",
				fmt.Sprintf("messages[%d]: %v", i, err))
			return
		}
		if len(mm) > 0 && !acceptsMedia {
			s.writeErr(w, http.StatusBadRequest, "invalid_request_error",
				fmt.Sprintf("model %q is text-only and cannot accept images", req.Model))
			return
		}
		media = append(media, mm...)
		msgs = append(msgs, engine.ChatMessage{Role: m.Role, Content: text})
	}

	// Reasoning is suppressed unless the caller asks for it. Left to itself the model
	// decides per request, and that decision is often a near-tie: the same prompt yields an
	// empty block on one run and hundreds of reasoning tokens on the next, which makes
	// throughput unpredictable as well as lower. A caller that wants reasoning asks for it.
	prompt, err := renderWithThinking(rend, msgs, req.Think)
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
		Media:     media,
		Speculate: req.Speculate,
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

// renderWithThinking renders a chat, suppressing the model's reasoning block unless the
// request asked for it. A model with no reasoning block renders unchanged, and a backend
// that cannot suppress one falls back to its ordinary rendering rather than failing.
func renderWithThinking(rend engine.Renderer, msgs []engine.ChatMessage, think *bool) (string, error) {
	tr, ok := rend.(engine.ThinkingRenderer)
	if !ok || tr == nil || !tr.SupportsThinking() {
		return rend.RenderChat(msgs)
	}
	return tr.RenderChatThinking(msgs, think != nil && *think)
}
