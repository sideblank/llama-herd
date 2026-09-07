// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
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
	// references hold each model's startup measurement, so a reader can tell a slow card or a
	// changed library from a slow engine without running anything.
	references map[string]any
	// libBench is llama-bench's reading of the same model, taken at startup.
	libBench       string
	samplerProfile func() *SamplerProfile
	sweep          json.RawMessage
	// host reads the machine's current state. Injectable because it is ambient: a test that
	// asserts on the warnings would otherwise depend on the load of whatever machine ran it,
	// passing when idle and failing under a parallel build. nil means read the real host.
	host func() hostinfo.Host
}

// SetHostReader overrides how the server reads machine state. For tests.
func (s *Server) SetHostReader(f func() hostinfo.Host) { s.host = f }

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
	Build       BuildInfo `json:"build"`
	Accelerated bool      `json:"accelerated"`
	Warning     string    `json:"warning,omitempty"`

	// LibraryBench is llama-bench's own reading of this model on this card, taken before
	// serving. It measures the library where selftest measures this engine, and the pair is
	// what tells a slow substrate from a herd that is not amortising. Absent unless asked
	// for, since it costs startup time.
	LibraryBench string `json:"library_bench,omitempty"`
	// Sampler is where time inside the sampler went. Absent unless the runtime supplies it.
	Sampler *SamplerProfile `json:"sampler,omitempty"`
	// Sweep is a configuration matrix measured before serving, verbatim as the sweep emitted
	// it. Absent unless one was run. Published here because the hosts this runs on give no
	// way to read a container's stdout.
	Sweep   json.RawMessage `json:"sweep,omitempty"`
	Host    hostinfo.Host   `json:"host"`
	Devices []DeviceInfo    `json:"devices"`
	Models  []ModelStatus   `json:"models"`
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

	// Selftest is what this deployment measured at startup, through the engine that serves
	// the traffic. It answers a question no serving metric does: whether the herd on THIS
	// card actually amortises, or whether every extra stream is costing a full pass while
	// the aggregate still looks plausible.
	Selftest any `json:"selftest,omitempty"`
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

// SamplerProfile splits time inside the sampler between narrowing the candidate set and
// running the library's chain over it.
//
// AvgCandidates is the check that matters: it says how many entries the chain was actually
// given. A value equal to the vocabulary means narrowing is not happening, however confidently
// the configuration suggests it should be — which is the difference between a measurement and
// an assumption.
type SamplerProfile struct {
	SelectMsPerToken float64 `json:"select_ms_per_token"`
	ApplyMsPerToken  float64 `json:"apply_ms_per_token"`
	AvgCandidates    float64 `json:"avg_candidates"`
	Calls            int64   `json:"calls"`
}

// WithSamplerProfile supplies a source for the sampler breakdown, read at request time so it
// reflects traffic rather than only startup. Passed as a function because the sampler lives
// behind cgo and this package must build without it.
func (s *Server) WithSamplerProfile(fn func() *SamplerProfile) *Server {
	s.samplerProfile = fn
	return s
}

// WithLibraryBench records llama-bench's own reading of this model, taken before serving.
//
// It measures the library rather than this engine, which is the comparison that says whether a
// slow deployment is slow underneath us or because of us. Reported verbatim: it is llama-bench's
// output, in the format published figures use, and reformatting it would only invite doubt
// about whether it is really that tool's number.
func (s *Server) WithLibraryBench(out string) *Server { s.libBench = out; return s }

// WithSweep records a configuration sweep taken before serving.
func (s *Server) WithSweep(raw []byte) *Server { s.sweep = raw; return s }

// WithSelftest records what a model measured at startup, for reporting on /v1/info.
//
// Set once during construction, like the placements beside it, and only read afterwards — so
// it needs no lock and must not be written while serving.
func (s *Server) WithSelftest(model string, st any) *Server {
	if s.references == nil {
		s.references = map[string]any{}
	}
	s.references[model] = st
	return s
}

// PlacementSource is implemented by a backend that can describe where its weights went.
type PlacementSource interface {
	Placement() Placement
}

// snapshot builds the current Info.
func (s *Server) snapshot() Info {
	var in Info
	in.Build = s.build
	if s.host != nil {
		in.Host = s.host()
	} else {
		in.Host = hostinfo.Read()
	}

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
		if st, ok := s.references[name]; ok {
			ms.Selftest = st
		}
		in.Models = append(in.Models, ms)
	}

	// Computed last, because the placement check needs the model list. Ordered by
	// severity: no accelerator at all, then an accelerator the weights never reached,
	// then a machine too busy to use what it has.
	in.LibraryBench = s.libBench
	in.Sweep = s.sweep
	if s.samplerProfile != nil {
		in.Sampler = s.samplerProfile()
	}

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
	toolsJSON, toolChoice := req.toolSpec()
	prompt, err := renderWithTools(rend, msgs, req.Think, toolsJSON, toolChoice)
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

	// think must be the value the RENDER used — renderWithTools computes it the same way.
	// Leaving it to the zero value would make every parse disagree with its own render, which
	// is the failure this field exists to prevent.
	tc := toolCtx{
		rend: rend, msgs: msgs, toolsJSON: toolsJSON, toolChoice: toolChoice,
		think: req.Think != nil && *req.Think,
	}
	if req.Stream {
		s.streamChat(w, req.Model, stream, tc)
		return
	}
	s.bufferChat(w, req.Model, stream, tc)
}

// maxRequestBytes bounds a request body. Prompts can be large — 128k of context is the
// point — so this is generous, but unbounded would let one caller exhaust memory.
const maxRequestBytes = 64 << 20

// toolCtx is what reading tool calls back out of a completion needs. Its zero value means
// "no tools were offered", which is the path every request took before tools existed.
type toolCtx struct {
	rend       engine.Renderer
	msgs       []engine.ChatMessage
	toolsJSON  string
	toolChoice string
	// think is the value the prompt was RENDERED with, carried so the parse can be given the
	// same one. The parser is derived from a re-render and carries its think tags; deriving it
	// from a different value reads the completion with the wrong grammar and finds no calls.
	think bool
}

func (s *Server) bufferChat(w http.ResponseWriter, model string, st *engine.Stream, tc toolCtx) {
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
	msg := respMessage{Role: "assistant", Content: sb.String()}
	// A model asked for tools may answer in prose instead, which is legitimate under "auto",
	// so no calls is not an error and leaves the reply unchanged. A parse FAILURE is different
	// and is reported: the reply still goes out with the raw generation, because dropping the
	// turn would be worse, but it no longer looks like the model simply declined.
	content, calls, err := parseToolCalls(tc, sb.String())
	switch {
	case err != nil:
		log.Printf("api: tool-call parse failed (%v) — returning the raw generation as content; "+
			"tool_calls will be absent and that is NOT the model declining", err)
	case len(calls) > 0:
		msg.Content = content
		msg.ToolCalls = calls
		fr = "tool_calls"
	}
	resp := chatResponse{
		ID: s.nextID(), Object: "chat.completion", Created: s.now().Unix(), Model: model,
		Choices: []choice{{
			Index:        0,
			Message:      &msg,
			FinishReason: &fr,
		}},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// streamChat writes server-sent events in the shape clients expect: a role-only first chunk,
// content deltas, a final chunk carrying finish_reason, then [DONE].
func (s *Server) streamChat(w http.ResponseWriter, model string, st *engine.Stream, tc toolCtx) {
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

	// Holds the answer while tools are in play; unused otherwise.
	var buffered strings.Builder
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
			// ⛔ A TOOLS REQUEST IS BUFFERED, NOT STREAMED AS PROSE. The calls a model asks
			// for arrive as markup its template defines, and forwarding that verbatim hands
			// the client `<tool_call>{...}` as assistant CONTENT — visible junk that is also
			// not the tool call it is supposed to be. The calls can only be recognised once
			// the text is whole, so the text is held until then.
			//
			// A request without tools streams exactly as before, token by token.
			if tc.toolsJSON != "" {
				buffered.WriteString(ev.Text)
			} else {
				send(choice{Index: 0, Delta: &respMessage{Content: ev.Text}})
			}
		}
		if ev.Done {
			reason = ev.Reason
		}
	}

	fr := finishReason(reason)
	if tc.toolsJSON != "" {
		content, calls, err := parseToolCalls(tc, buffered.String())
		switch {
		case err != nil:
			// Report it and release what was held. The turn is never silently dropped, and
			// the absent tool_calls no longer reads as the model choosing prose.
			log.Printf("api: tool-call parse failed on the streaming path (%v) — releasing the "+
				"raw generation; tool_calls will be absent and that is NOT the model declining", err)
			send(choice{Index: 0, Delta: &respMessage{Content: buffered.String()}})
		case len(calls) > 0:
			send(choice{Index: 0, Delta: &respMessage{Content: content, ToolCalls: calls}})
			fr = "tool_calls"
		default:
			// The model answered in prose, which is legitimate under "auto". Release what
			// was held so a buffered turn is never silently dropped.
			send(choice{Index: 0, Delta: &respMessage{Content: buffered.String()}})
		}
	}
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

// renderWithTools renders a chat, presenting tool definitions when the request carried any.
//
// A request with no tools takes exactly the path it took before tools existed. A request WITH
// tools that this backend cannot present is refused rather than served without them: answering
// it anyway returns a confident reply to a question the model was never asked, and the caller
// has no way to tell that from a model that considered the tools and declined.
func renderWithTools(rend engine.Renderer, msgs []engine.ChatMessage, think *bool, toolsJSON, toolChoice string) (string, error) {
	if toolsJSON == "" {
		return renderWithThinking(rend, msgs, think)
	}
	tr, ok := rend.(engine.ToolRenderer)
	if !ok || tr == nil || !tr.SupportsTools() {
		return "", errors.New("this model cannot be given tool definitions")
	}
	return tr.RenderChatTools(msgs, toolsJSON, toolChoice, think != nil && *think)
}

// errNoToolSupport reports that tools were requested of a backend that cannot render them.
var errNoToolSupport = errors.New("api: this backend cannot parse tool calls")

// parseToolCalls reads a completion back for the calls the model asked for.
//
// ⛔ THREE OUTCOMES, DELIBERATELY DISTINGUISHABLE. A nil error with no calls means the model
// was asked and chose prose, which is legitimate under "auto". A non-nil error means the parse
// itself failed and NOTHING can be concluded about what the model wanted. Collapsing those two
// into one boolean is how a broken parse looks exactly like a model declining: the caller gets
// a 200, the raw generation lands in `content`, and nobody learns the parse failed. That is not
// hypothetical — a fleet card was observed emitting a complete, well-formed tool call as text
// while the API reported no calls, and this discard is why it took an experiment to see.
//
// Callers decide what to do with a failure; they must not treat it as "no calls".
func parseToolCalls(tc toolCtx, text string) (string, []respToolCall, error) {
	if tc.toolsJSON == "" {
		return text, nil, nil
	}
	tr, ok := tc.rend.(engine.ToolRenderer)
	if !ok || tr == nil {
		// The request carried tools and this backend cannot render them. Saying so is the
		// point: answering anyway would be a confident reply to a request never served.
		return text, nil, errNoToolSupport
	}
	content, calls, err := tr.ParseChatOutput(tc.msgs, tc.toolsJSON, tc.toolChoice, text, tc.think)
	if err != nil {
		return text, nil, err
	}
	if len(calls) == 0 {
		// ⛔ THE LIBRARY FINDING NOTHING IS NOT PROOF THE MODEL DECLINED. llama.cpp's grammar for
		// the Qwen XML form permutes REQUIRED parameters but requires optional ones to follow all
		// of them, while the model emits parameters in schema-declaration order — so a call whose
		// first parameter is optional matches nothing. Present in every revision through the
		// current master, so pinning forward does not avoid it.
		//
		// When the library finds nothing but the completion carries a well-formed block, read the
		// format the template documents. Reported as a distinct outcome by the caller, so the two
		// parsers can never silently disagree: exactly one answers, and the response says which.
		if native, rest := parseToolXML(text); len(native) > 0 {
			for i := range native {
				if native[i].ID == "" {
					native[i].ID = fmt.Sprintf("call_%d", i)
				}
			}
			// Announced, never silent — if these two ever disagree about a call, the log says
			// which one produced it. Not an error: the calls are real and the request succeeded.
			log.Printf("api: llama.cpp parsed no tool calls but the completion carries %d "+
				"well-formed one(s); using the template-documented format", len(native))
			return rest, native, nil
		}
	}
	if len(calls) == 0 {
		return text, nil, nil
	}
	out := make([]respToolCall, 0, len(calls))
	for i, c := range calls {
		id := c.ID
		if id == "" {
			id = fmt.Sprintf("call_%d", i)
		}
		out = append(out, respToolCall{
			ID: id, Type: "function",
			Function: respToolFunction{Name: c.Name, Arguments: c.Arguments},
		})
	}
	return content, out, nil
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
