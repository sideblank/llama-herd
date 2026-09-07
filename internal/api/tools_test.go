// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package api

import "testing"

// A request's tools decide whether the tool path runs at all, so the cases that must NOT
// enable it matter as much as the ones that must. Enabling it on an empty list would present
// an empty tool block to every model; failing to enable it on a real list is the bug this
// whole change exists to fix.
func TestToolSpec(t *testing.T) {
	const one = `[{"type":"function","function":{"name":"lookup","parameters":{}}}]`

	for _, tc := range []struct {
		name       string
		tools      string
		choice     string
		wantTools  bool
		wantChoice string
	}{
		{name: "absent", tools: "", choice: "", wantTools: false},
		{name: "empty array", tools: `[]`, choice: "", wantTools: false},
		{name: "malformed", tools: `{"not":"an array"}`, choice: "", wantTools: false},
		// A choice with nothing to choose from is a caller error, not a request for tools.
		{name: "choice alone", tools: "", choice: `"required"`, wantTools: false},
		{name: "default choice", tools: one, choice: "", wantTools: true, wantChoice: "auto"},
		{name: "required", tools: one, choice: `"required"`, wantTools: true, wantChoice: "required"},
		{name: "none", tools: one, choice: `"none"`, wantTools: true, wantChoice: "none"},
		// Naming one function is a required call; the model still picks from the list.
		{name: "named function", tools: one,
			choice:    `{"type":"function","function":{"name":"lookup"}}`,
			wantTools: true, wantChoice: "required"},
		// An unrecognised word must not be passed through to the template.
		{name: "unknown choice", tools: one, choice: `"whatever"`, wantTools: true, wantChoice: "auto"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var r ChatRequest
			if tc.tools != "" {
				r.Tools = []byte(tc.tools)
			}
			if tc.choice != "" {
				r.ToolChoice = []byte(tc.choice)
			}
			gotTools, gotChoice := r.toolSpec()
			if (gotTools != "") != tc.wantTools {
				t.Fatalf("tools enabled = %v, want %v (got %q)", gotTools != "", tc.wantTools, gotTools)
			}
			if tc.wantTools && gotChoice != tc.wantChoice {
				t.Fatalf("choice = %q, want %q", gotChoice, tc.wantChoice)
			}
		})
	}
}
