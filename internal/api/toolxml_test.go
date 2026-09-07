// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"encoding/json"
	"strings"
	"testing"
)

// The exact bytes a live Qwen3.5 model returned under tool_choice=required, which llama.cpp
// parsed as zero calls. Testing against a real generation rather than a hand-written sample is
// the point: a parser checked only against what its author imagines the model emits is how this
// class of bug survives — the format looks right to everyone who reads it and matches nothing.
const liveQwenGeneration = "<tool_call>\n<function=lookup_indicator>\n<parameter=country>\n" +
	"Germany\n</parameter>\n<parameter=indicator>\ninflation rate\n</parameter>\n</function>\n</tool_call>"

func TestParsesARealQwenToolCall(t *testing.T) {
	calls, content := parseToolXML(liveQwenGeneration)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Function.Name != "lookup_indicator" {
		t.Errorf("name = %q", calls[0].Function.Name)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &args); err != nil {
		t.Fatalf("arguments must be a JSON object: %v", err)
	}
	if args["country"] != "Germany" || args["indicator"] != "inflation rate" {
		t.Errorf("arguments = %v", args)
	}
	if content != "" {
		t.Errorf("content should be empty once consumed, got %q", content)
	}
}

// ⛔ A parser that cannot decline is worse than none: it turns an ordinary answer into a
// fabricated tool invocation the caller will act on.
func TestOrdinaryTextIsNeverAToolCall(t *testing.T) {
	for _, s := range []string{
		"Germany's inflation rate is about 2.2% as of last month.",
		"You could use <tool_call> for that",
		"<tool_call>\n</tool_call>",
		"<tool_call>\n<function=>\n</function>\n</tool_call>",
	} {
		calls, content := parseToolXML(s)
		if len(calls) != 0 {
			t.Errorf("%q produced %d calls, want 0", s, len(calls))
		}
		if content == "" {
			t.Errorf("%q lost its text — a non-call must survive as content", s)
		}
	}
}

// A truncated block is not a call, and its text must survive: otherwise a max_tokens cut is
// indistinguishable from a model that said nothing.
func TestTruncatedCallStaysText(t *testing.T) {
	calls, content := parseToolXML("<tool_call>\n<function=f>\n<parameter=x>\nval")
	if len(calls) != 0 {
		t.Fatalf("truncated block must not parse as a call, got %d", len(calls))
	}
	if !strings.Contains(content, "<function=f>") {
		t.Errorf("truncated text must survive, got %q", content)
	}
}

func TestProseAroundCallsAndMultipleCalls(t *testing.T) {
	in := "Checking.\n" + liveQwenGeneration +
		"\n<tool_call>\n<function=second>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"
	calls, content := parseToolXML(in)
	if len(calls) != 2 {
		t.Fatalf("expected 2 calls, got %d", len(calls))
	}
	if calls[1].Function.Name != "second" {
		t.Errorf("second call name = %q", calls[1].Function.Name)
	}
	if !strings.Contains(content, "Checking.") {
		t.Errorf("prose must be preserved, got %q", content)
	}
}

// The template renders non-scalar arguments as JSON, so they must come back as structure.
func TestStructuredArgumentsDecode(t *testing.T) {
	calls, _ := parseToolXML(
		"<tool_call>\n<function=f>\n<parameter=items>\n[1,2,3]\n</parameter>\n</function>\n</tool_call>")
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &args); err != nil {
		t.Fatal(err)
	}
	if arr, ok := args["items"].([]any); !ok || len(arr) != 3 {
		t.Errorf("items should decode as a 3-element array, got %#v", args["items"])
	}
}

// ⛔ The reason this parser exists: the optional-parameter-first ordering llama.cpp's grammar
// cannot match. If this ever fails, the workaround has stopped covering the case it was built
// for and someone will conclude the model declined.
func TestOptionalParameterFirstStillParses(t *testing.T) {
	// `country` is optional in the schema and the model emitted it first; llama.cpp's grammar
	// permutes required args but requires optionals to follow all of them.
	calls, _ := parseToolXML(liveQwenGeneration)
	if len(calls) != 1 {
		t.Fatalf("the ordering this exists to handle no longer parses: %d calls", len(calls))
	}
}
