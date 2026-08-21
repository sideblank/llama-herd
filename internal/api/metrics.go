// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"fmt"
	"net/http"
	"strings"
)

// metrics serves the Prometheus text exposition format.
//
// Written by hand rather than pulled from a client library: the whole surface is a few
// counters and gauges, and a monitoring endpoint is not worth a dependency tree in a binary
// people run on their own machines.
//
// Counters end in _total and only increase; gauges move both ways. Names are prefixed
// llama_herd_ so they do not collide in a shared scrape.
func (s *Server) metrics(w http.ResponseWriter, _ *http.Request) {
	in := s.snapshot()
	var b strings.Builder

	g := func(name, help string, value float64, labels ...string) {
		emit(&b, "gauge", name, help, value, labels...)
	}
	c := func(name, help string, value float64, labels ...string) {
		emit(&b, "counter", name, help, value, labels...)
	}

	// Build identity as a labelled constant, the conventional way to expose version info.
	fmt.Fprintf(&b, "# HELP llama_herd_build_info Build identity of the running binary.\n")
	fmt.Fprintf(&b, "# TYPE llama_herd_build_info gauge\n")
	fmt.Fprintf(&b, "llama_herd_build_info{version=%q,commit=%q,llama_cpp=%q} 1\n",
		esc(in.Build.Version), esc(in.Build.Commit), esc(in.Build.LlamaCppRef))

	g("llama_herd_accelerated", "1 when a dedicated-memory GPU was found, 0 when running on CPU.",
		boolVal(in.Accelerated))

	g("llama_herd_host_cpus", "Logical CPUs visible to the process.", float64(in.Host.CPUs))
	g("llama_herd_host_load1", "One-minute load average. 0 where unavailable.", in.Host.LoadAvg1)
	g("llama_herd_host_load5", "Five-minute load average.", in.Host.LoadAvg5)
	g("llama_herd_host_load15", "Fifteen-minute load average.", in.Host.LoadAvg15)
	g("llama_herd_host_mem_total_bytes", "Total system memory.", float64(in.Host.MemTotalBytes))
	g("llama_herd_host_mem_available_bytes", "Available system memory.", float64(in.Host.MemAvailableBytes))
	g("llama_herd_process_heap_bytes", "Go heap in use.", float64(in.Host.ProcHeapBytes))
	g("llama_herd_process_goroutines", "Live goroutines.", float64(in.Host.Goroutines))
	g("llama_herd_uptime_seconds", "Seconds since start.", in.Host.UptimeSeconds)

	for _, d := range in.Devices {
		lbl := []string{"device", fmt.Sprint(d.Index), "name", d.Name, "type", d.Type}
		g("llama_herd_device_memory_total_bytes", "Device memory.", float64(d.TotalBytes), lbl...)
		// Free VRAM is the number that decides whether another model or more context
		// will fit, and it moves as other processes allocate.
		g("llama_herd_device_memory_free_bytes", "Device memory currently free.", float64(d.FreeBytes), lbl...)
	}

	for _, m := range in.Models {
		lbl := []string{"model", m.Name}
		g("llama_herd_model_up", "1 when the model's decode loop is running.", boolVal(m.OK), lbl...)
		if m.Stats == nil {
			continue
		}
		st := m.Stats
		g("llama_herd_streams_max", "Concurrent generations this model can serve.", float64(st.StreamsMax), lbl...)
		g("llama_herd_streams_active", "Generations decoding now.", float64(st.StreamsActive), lbl...)
		g("llama_herd_queue_depth", "Requests waiting for a stream. Persistently above zero means saturated.",
			float64(st.Queued), lbl...)
		g("llama_herd_context_total_tokens", "Context shared across streams.", float64(st.ContextTotal), lbl...)
		g("llama_herd_context_per_stream_tokens", "Context available to one stream.", float64(st.ContextPerStream), lbl...)
		c("llama_herd_requests_total", "Requests accepted.", float64(st.RequestsTotal), lbl...)
		c("llama_herd_requests_failed_total", "Requests that ended in error.", float64(st.RequestsFailed), lbl...)
		c("llama_herd_tokens_generated_total", "Tokens generated.", float64(st.TokensGenerated), lbl...)
		c("llama_herd_prompt_tokens_total", "Prompt tokens accepted.", float64(st.PromptTokens), lbl...)
		c("llama_herd_drafts_proposed_total", "Speculative tokens offered to the target.",
			float64(st.DraftsProposed), lbl...)
		c("llama_herd_drafts_accepted_total", "Speculative tokens the target kept. The ratio to proposed is the acceptance rate, unaffected by prompt length, stream count or host contention.",
			float64(st.DraftsAccepted), lbl...)
		g("llama_herd_draft_acceptance_rate", "Fraction of proposed draft tokens accepted. Zero with no drafter; proposals with no acceptances means batch space spent for nothing.",
			st.AcceptanceRate(), lbl...)
		c("llama_herd_evictions_total", "Streams ended because the KV cache filled. A rising count means the context budget is over-committed.",
			float64(st.EvictionsTotal), lbl...)
	}

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_, _ = w.Write([]byte(b.String()))
}

// emitted tracks which metric names have had their HELP and TYPE written, since those must
// appear once per name even when the metric is repeated across label sets.
func emit(b *strings.Builder, kind, name, help string, value float64, labels ...string) {
	key := "\n# TYPE " + name + " "
	if !strings.Contains(b.String(), key) {
		fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, kind)
	}
	if len(labels) == 0 {
		fmt.Fprintf(b, "%s %g\n", name, value)
		return
	}
	var pairs []string
	for i := 0; i+1 < len(labels); i += 2 {
		pairs = append(pairs, fmt.Sprintf("%s=%q", labels[i], esc(labels[i+1])))
	}
	fmt.Fprintf(b, "%s{%s} %g\n", name, strings.Join(pairs, ","), value)
}

func boolVal(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// esc removes the characters that would break the exposition format.
func esc(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", " ")
	return r.Replace(s)
}
