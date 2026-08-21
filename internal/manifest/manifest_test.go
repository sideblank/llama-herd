// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package manifest

import (
	"strings"
	"testing"
)

func parse(t *testing.T, s string) (*Manifest, error) {
	t.Helper()
	return Parse(strings.NewReader(s))
}

func TestMinimalManifestParsesWithDefaults(t *testing.T) {
	m, err := parse(t, `{"models":[{"name":"a","path":"/a.gguf","context":4096}]}`)
	if err != nil {
		t.Fatal(err)
	}
	if m.Listen != ":8080" {
		t.Errorf("listen = %q, want :8080", m.Listen)
	}
	mm := m.Models[0]
	if mm.Streams != 1 {
		t.Errorf("streams = %d, want 1", mm.Streams)
	}
	if mm.Batch == 0 {
		t.Error("batch should default to non-zero")
	}
	if mm.SplitMode != SplitNone {
		t.Errorf("split_mode = %q, want none", mm.SplitMode)
	}
}

// An operator-authored config must fail loudly on a typo rather than silently leaving the
// setting at its default.
func TestUnknownKeyIsRejected(t *testing.T) {
	_, err := parse(t, `{"models":[{"name":"a","path":"/a.gguf","context":4096,"stremas":8}]}`)
	if err == nil {
		t.Fatal("a misspelled key should be rejected")
	}
}

func TestDuplicateNamesRejected(t *testing.T) {
	_, err := parse(t, `{"models":[
		{"name":"a","path":"/a.gguf","context":4096},
		{"name":"a","path":"/b.gguf","context":4096}]}`)
	if err == nil || !strings.Contains(err.Error(), "duplicate name") {
		t.Fatalf("want duplicate-name error, got %v", err)
	}
}

func TestRequiredFields(t *testing.T) {
	cases := map[string]string{
		"name":    `{"models":[{"path":"/a.gguf","context":4096}]}`,
		"path":    `{"models":[{"name":"a","context":4096}]}`,
		"context": `{"models":[{"name":"a","path":"/a.gguf"}]}`,
		"models":  `{"models":[]}`,
	}
	for want, body := range cases {
		if _, err := parse(t, body); err == nil {
			t.Errorf("missing %s should be rejected", want)
		}
	}
}

func TestInvalidSplitModeRejected(t *testing.T) {
	_, err := parse(t, `{"models":[{"name":"a","path":"/a.gguf","context":4096,"split_mode":"sideways"}]}`)
	if err == nil || !strings.Contains(err.Error(), "split_mode") {
		t.Fatalf("want split_mode error, got %v", err)
	}
}

// tensor_split silently does nothing when the model is not being split, which is the kind
// of misconfiguration that looks like it worked.
func TestTensorSplitWithoutSplitModeIsRejected(t *testing.T) {
	_, err := parse(t, `{"models":[{"name":"a","path":"/a.gguf","context":4096,
		"tensor_split":[0.5,0.5]}]}`)
	if err == nil || !strings.Contains(err.Error(), "no effect") {
		t.Fatalf("want a warning that tensor_split does nothing, got %v", err)
	}
}

func TestBatchSmallerThanStreamsRejected(t *testing.T) {
	_, err := parse(t, `{"models":[{"name":"a","path":"/a.gguf","context":65536,
		"streams":16,"batch":8}]}`)
	if err == nil || !strings.Contains(err.Error(), "cannot decode in one pass") {
		t.Fatalf("want batch/streams error, got %v", err)
	}
}

func TestContextSpreadTooThinRejected(t *testing.T) {
	_, err := parse(t, `{"models":[{"name":"a","path":"/a.gguf","context":1024,"streams":16}]}`)
	if err == nil || !strings.Contains(err.Error(), "too little to be useful") {
		t.Fatalf("want per-stream context error, got %v", err)
	}
}

// All problems should be reported at once so a broken manifest takes one edit to fix.
func TestValidationReportsEveryProblem(t *testing.T) {
	_, err := parse(t, `{"models":[{"name":"","path":"","context":0}]}`)
	if err == nil {
		t.Fatal("expected errors")
	}
	for _, want := range []string{"name is required", "path is required", "context is required"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error should mention %q; got:\n%s", want, err)
		}
	}
}

func TestSamplingZeroIsDistinguishableFromUnset(t *testing.T) {
	m, err := parse(t, `{"models":[{"name":"a","path":"/a.gguf","context":4096,
		"sampling":{"temperature":0}}]}`)
	if err != nil {
		t.Fatal(err)
	}
	s := m.Models[0].Sampling
	if s.Temperature == nil {
		t.Fatal("an explicit temperature of 0 must not read as unset — it means greedy")
	}
	if *s.Temperature != 0 {
		t.Fatalf("temperature = %v, want 0", *s.Temperature)
	}

	m2, _ := parse(t, `{"models":[{"name":"a","path":"/a.gguf","context":4096}]}`)
	if m2.Models[0].Sampling.Temperature != nil {
		t.Fatal("an omitted temperature should be nil, not 0")
	}
}

// MTP drafting reads from a prediction head that load_mtp is what makes resident. Asking for
// the draft source without the head is a configuration that starts, serves, and never
// speculates — the failure is invisible unless the manifest refuses it.
func TestMTPSpeculationWithoutLoadMTPRejected(t *testing.T) {
	_, err := parse(t, `{"models":[{"name":"a","path":"/a.gguf","context":4096,
		"speculation":{"type":"mtp"}}]}`)
	if err == nil || !strings.Contains(err.Error(), "load_mtp") {
		t.Fatalf("want load_mtp error, got %v", err)
	}
}

func TestMTPSpeculationWithLoadMTPAccepted(t *testing.T) {
	m, err := parse(t, `{"models":[{"name":"a","path":"/a.gguf","context":4096,
		"load_mtp":true,"speculation":{"type":"mtp","max_draft":4}}]}`)
	if err != nil {
		t.Fatalf("want accepted, got %v", err)
	}
	if got := m.Models[0].Speculation.Type; got != "mtp" {
		t.Fatalf("speculation type = %q, want mtp", got)
	}
}

func TestUnknownSpeculationTypeRejected(t *testing.T) {
	_, err := parse(t, `{"models":[{"name":"a","path":"/a.gguf","context":4096,
		"speculation":{"type":"oracle"}}]}`)
	if err == nil || !strings.Contains(err.Error(), "oracle") {
		t.Fatalf("want type error naming the value, got %v", err)
	}
}
