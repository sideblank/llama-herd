// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/sideblank/llama-herd/internal/engine"
	"github.com/sideblank/llama-herd/internal/enginetest"
)

// echoRenderer flattens messages the way a trivial template would.
type echoRenderer struct{}

func (echoRenderer) RenderChat(msgs []engine.ChatMessage) (string, error) {
	var sb strings.Builder
	for _, m := range msgs {
		sb.WriteString(m.Role)
		sb.WriteString(": ")
		sb.WriteString(m.Content)
		sb.WriteString("\n")
	}
	return sb.String(), nil
}

func newTestServer(t *testing.T, script string) (*httptest.Server, context.CancelFunc) {
	t.Helper()
	f := enginetest.New(4, 32, script)
	e := engine.New(f, engine.Config{})
	reg := engine.NewRegistry()
	if err := reg.Add("test-model", e, echoRenderer{}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	reg.Start(ctx)

	srv := New(reg)
	srv.now = func() time.Time { return time.Unix(1700000000, 0) }
	ts := httptest.NewServer(srv.Handler())
	return ts, func() { ts.Close(); cancel() }
}

func post(t *testing.T, ts *httptest.Server, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestChatCompletionNonStreaming(t *testing.T) {
	ts, done := newTestServer(t, "hello")
	defer done()

	resp := post(t, ts, `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	var got chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Object != "chat.completion" {
		t.Fatalf("object = %q", got.Object)
	}
	if len(got.Choices) != 1 {
		t.Fatalf("choices = %d", len(got.Choices))
	}
	if c := got.Choices[0]; c.Message == nil || c.Message.Content != "hello" {
		t.Fatalf("content = %+v, want %q", c.Message, "hello")
	}
	if got.Choices[0].FinishReason == nil || *got.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason = %v", got.Choices[0].FinishReason)
	}
}

func TestChatCompletionStreaming(t *testing.T) {
	ts, done := newTestServer(t, "abc")
	defer done()

	resp := post(t, ts, `{"model":"test-model","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type = %q", ct)
	}

	var content strings.Builder
	var sawRole, sawDone bool
	var finish string

	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			sawDone = true
			break
		}
		var chunk chatResponse
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("bad chunk %q: %v", payload, err)
		}
		if chunk.Object != "chat.completion.chunk" {
			t.Fatalf("object = %q", chunk.Object)
		}
		for _, c := range chunk.Choices {
			if c.Delta == nil {
				continue
			}
			if c.Delta.Role == "assistant" {
				sawRole = true
			}
			content.WriteString(c.Delta.Content)
			if c.FinishReason != nil {
				finish = *c.FinishReason
			}
		}
	}

	if !sawRole {
		t.Error("no role-bearing first chunk")
	}
	if !sawDone {
		t.Error("stream did not terminate with [DONE]")
	}
	if content.String() != "abc" {
		t.Errorf("content = %q, want %q", content.String(), "abc")
	}
	if finish != "stop" {
		t.Errorf("finish_reason = %q, want %q", finish, "stop")
	}
}

func TestUnknownModelIs404(t *testing.T) {
	ts, done := newTestServer(t, "x")
	defer done()

	resp := post(t, ts, `{"model":"nope","messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", resp.StatusCode)
	}
	var e apiError
	if err := json.NewDecoder(resp.Body).Decode(&e); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(e.Error.Message, "nope") {
		t.Fatalf("error should name the model, got %q", e.Error.Message)
	}
}

// Clients legitimately send fields this server does not implement. Rejecting them would
// break callers over something they cannot control.
func TestUnknownFieldsAreIgnoredNotRejected(t *testing.T) {
	ts, done := newTestServer(t, "ok")
	defer done()

	resp := post(t, ts, `{
		"model":"test-model",
		"messages":[{"role":"user","content":"hi"}],
		"temperature":0.7, "top_p":0.9, "presence_penalty":0.1,
		"tools":[], "user":"someone", "seed":42
	}`)
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200 — unknown fields must not be rejected", resp.StatusCode)
	}
}

// "stop" is a string in some clients and an array in others.
func TestStopAcceptsStringOrArray(t *testing.T) {
	for _, body := range []string{
		`{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stop":"X"}`,
		`{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stop":["X","Y"]}`,
	} {
		ts, done := newTestServer(t, "aXb")
		resp := post(t, ts, body)
		if resp.StatusCode != 200 {
			t.Fatalf("status = %d for body %s", resp.StatusCode, body)
		}
		resp.Body.Close()
		done()
	}
}

func TestMissingModelAndMessagesRejected(t *testing.T) {
	ts, done := newTestServer(t, "x")
	defer done()

	for _, body := range []string{
		`{"messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"test-model","messages":[]}`,
		`{"model":"test-model"}`,
	} {
		resp := post(t, ts, body)
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("status = %d for %s, want 400", resp.StatusCode, body)
		}
		resp.Body.Close()
	}
}

func TestListModels(t *testing.T) {
	ts, done := newTestServer(t, "x")
	defer done()

	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var list modelList
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		t.Fatal(err)
	}
	if list.Object != "list" || len(list.Data) != 1 || list.Data[0].ID != "test-model" {
		t.Fatalf("unexpected model list: %+v", list)
	}
}

func TestHealth(t *testing.T) {
	ts, done := newTestServer(t, "x")
	defer done()

	resp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
}

func TestMaxTokensBothSpellings(t *testing.T) {
	ts, done := newTestServer(t, "abcdef")
	defer done()

	for _, body := range []string{
		`{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":2}`,
		`{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":2}`,
	} {
		resp := post(t, ts, body)
		var got chatResponse
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if len(got.Choices) != 1 || got.Choices[0].Message == nil {
			t.Fatalf("no choice for %s", body)
		}
		if n := len(got.Choices[0].Message.Content); n != 2 {
			t.Errorf("content %q has %d chars, want 2 (%s)", got.Choices[0].Message.Content, n, body)
		}
		if fr := got.Choices[0].FinishReason; fr == nil || *fr != "length" {
			t.Errorf("finish_reason = %v, want length", fr)
		}
	}
}

// Sampling fields must reach the engine rather than being silently dropped, which was the
// behaviour before per-request chains existed.
func TestSamplingFieldsReachTheEngine(t *testing.T) {
	ts, done := newTestServer(t, "hi")
	defer done()

	resp := post(t, ts, `{"model":"test-model","messages":[{"role":"user","content":"hi"}],
		"temperature":0.2,"top_p":0.5,"seed":7}`)
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
}

func TestSamplingRequestMapping(t *testing.T) {
	var req ChatRequest
	body := `{"model":"m","messages":[],"temperature":0,"top_k":10,"presence_penalty":0.5}`
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	p := req.sampling()
	if p == nil {
		t.Fatal("sampling should not be nil when fields are present")
	}
	// An explicit zero temperature means greedy and must survive as a set value.
	if p.Temperature == nil || *p.Temperature != 0 {
		t.Fatalf("temperature = %v, want an explicit 0", p.Temperature)
	}
	if p.TopK == nil || *p.TopK != 10 {
		t.Fatalf("top_k = %v", p.TopK)
	}
	if p.PresencePenalty == nil || *p.PresencePenalty != 0.5 {
		t.Fatalf("presence_penalty = %v", p.PresencePenalty)
	}
	if p.TopP != nil {
		t.Fatal("omitted top_p should stay nil so the model default survives")
	}
}

func TestNoSamplingFieldsMeansNoOverride(t *testing.T) {
	var req ChatRequest
	if err := json.Unmarshal([]byte(`{"model":"m","messages":[]}`), &req); err != nil {
		t.Fatal(err)
	}
	if p := req.sampling(); p != nil {
		t.Fatalf("expected nil override, got %+v", p)
	}
}

// A GPU runtime that silently falls back to CPU answers correctly and reports itself
// healthy. The only cheap way to tell is to ask what hardware it found.
func TestInfoReportsHardwareAndWarnsOnCPUOnly(t *testing.T) {
	ts, done := newTestServer(t, "hi")
	defer done()

	resp, err := http.Get(ts.URL + "/v1/info")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var got Info
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Accelerated {
		t.Error("no devices reported, so accelerated must be false")
	}
	if got.Warning == "" {
		t.Error("a CPU-only process must say so; silence is the failure mode this prevents")
	}
	if len(got.Models) != 1 || got.Models[0].Name != "test-model" {
		t.Errorf("models = %+v", got.Models)
	}
	if got.Host.CPUs < 1 {
		t.Error("host stats should be populated")
	}
	if got.Models[0].Stats == nil {
		t.Fatal("per-model stats should be present")
	}
	if got.Models[0].Stats.StreamsMax != 4 {
		t.Errorf("streams_max = %d, want 4", got.Models[0].Stats.StreamsMax)
	}
}

func TestInfoReportsGPUWhenPresent(t *testing.T) {
	f := enginetest.New(2, 32, "hi")
	reg := engine.NewRegistry()
	if err := reg.Add("m", engine.New(f, engine.Config{}), echoRenderer{}); err != nil {
		t.Fatal(err)
	}
	srv := New(reg).
		WithBuild(BuildInfo{Version: "v1", LlamaCppRef: "b10545"}).
		WithDevices(func() []DeviceInfo {
			return []DeviceInfo{{Index: 0, Name: "Test GPU", Type: "gpu", TotalBytes: 24 << 30}}
		})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/v1/info")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var got Info
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.Accelerated {
		t.Error("a gpu device must mark the process accelerated")
	}
	if got.Warning != "" {
		t.Errorf("no warning expected when a GPU is present, got %q", got.Warning)
	}
	if got.Build.LlamaCppRef != "b10545" {
		t.Errorf("build info missing: %+v", got.Build)
	}
}

func TestMetricsExposition(t *testing.T) {
	ts, done := newTestServer(t, "hello")
	defer done()

	// Generate some work so the counters are non-zero.
	post(t, ts, `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`).Body.Close()

	resp, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Fatalf("content-type = %q", ct)
	}
	b, _ := io.ReadAll(resp.Body)
	body := string(b)

	for _, want := range []string{
		"llama_herd_build_info{",
		"llama_herd_accelerated",
		"llama_herd_host_cpus",
		`llama_herd_streams_max{model="test-model"}`,
		`llama_herd_queue_depth{model="test-model"}`,
		"llama_herd_requests_total",
		"llama_herd_tokens_generated_total",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics missing %q", want)
		}
	}

	// Every metric needs exactly one HELP and one TYPE line, even when repeated across
	// label sets — a scrape rejects duplicates.
	for _, name := range []string{"llama_herd_streams_max", "llama_herd_requests_total"} {
		if n := strings.Count(body, "# TYPE "+name+" "); n != 1 {
			t.Errorf("%s has %d TYPE lines, want exactly 1", name, n)
		}
		if n := strings.Count(body, "# HELP "+name+" "); n != 1 {
			t.Errorf("%s has %d HELP lines, want exactly 1", name, n)
		}
	}

	// Counters must have recorded the request we just made.
	if !strings.Contains(body, `llama_herd_requests_total{model="test-model"} 1`) {
		t.Errorf("request counter did not increment:\n%s", body)
	}
}

// A device being present is not the same as the model running on it. Offload can be
// configured off or fail, and the server then reports an accelerator while working on CPU —
// which reads as a slow GPU rather than an unused one.
func TestGPUPresentButModelOnCPUIsWarned(t *testing.T) {
	f := enginetest.New(2, 32, "hi")
	reg := engine.NewRegistry()
	if err := reg.Add("m", engine.New(f, engine.Config{}), echoRenderer{}); err != nil {
		t.Fatal(err)
	}
	srv := New(reg).
		WithDevices(func() []DeviceInfo {
			return []DeviceInfo{{Index: 0, Name: "GPU", Type: "gpu", TotalBytes: 24 << 30}}
		}).
		WithPlacement("m", func() Placement {
			return Placement{GPULayersRequested: 0, LayersTotal: 32, OnGPU: false}
		})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/v1/info")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var got Info
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.Accelerated {
		t.Error("a GPU device is present, so accelerated should be true")
	}
	if got.Warning == "" || !strings.Contains(got.Warning, "running on CPU") {
		t.Fatalf("a GPU present with the model on CPU must warn; got %q", got.Warning)
	}
	if got.Models[0].Placement == nil || got.Models[0].Placement.OnGPU {
		t.Error("placement should report the model is not on the GPU")
	}
}

func TestModelOnGPUProducesNoWarning(t *testing.T) {
	f := enginetest.New(2, 32, "hi")
	reg := engine.NewRegistry()
	if err := reg.Add("m", engine.New(f, engine.Config{}), echoRenderer{}); err != nil {
		t.Fatal(err)
	}
	srv := New(reg).
		WithDevices(func() []DeviceInfo {
			return []DeviceInfo{{Index: 0, Name: "GPU", Type: "gpu", TotalBytes: 24 << 30}}
		}).
		WithPlacement("m", func() Placement {
			return Placement{GPULayersRequested: -1, LayersTotal: 32, OnGPU: true}
		})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, _ := http.Get(ts.URL + "/v1/info")
	defer resp.Body.Close()
	var got Info
	_ = json.NewDecoder(resp.Body).Decode(&got)
	if got.Warning != "" {
		t.Fatalf("no warning expected when the model is on the GPU, got %q", got.Warning)
	}
}
