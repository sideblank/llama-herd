// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"strings"
	"testing"
)

const sampleJSON = `[
  {"file_path": "auth/user.go", "symbol_type": "struct", "symbol_name": "User", "is_exported": true},
  {"file_path": "auth/user.go", "symbol_type": "struct", "symbol_name": "Session", "is_exported": true}
]`

func TestFlattenProducesOneRecordPerElement(t *testing.T) {
	recs, err := FlattenJSONArray(strings.NewReader(sampleJSON), DefaultJSONCanon())
	if err != nil {
		t.Fatal(err)
	}
	if len(recs) != 2 {
		t.Fatalf("want 2 records, got %d", len(recs))
	}
	out := DefaultJSONCanon().Render(recs)
	if !strings.Contains(out, "symbol_name:User") || !strings.Contains(out, "is_exported:1") {
		t.Fatalf("unexpected render:\n%s", out)
	}
	if strings.Count(out, "\n") != 1 {
		t.Fatalf("two records means one separator, got %q", out)
	}
}

func TestOriginalIndexIsPreserved(t *testing.T) {
	recs, _ := FlattenJSONArray(strings.NewReader(sampleJSON), DefaultJSONCanon())
	if recs[1].Index != 1 {
		t.Fatal("flattening is lossy — an assertion naming a bad field needs a way back to the source element")
	}
}

func TestFieldOrderIsStable(t *testing.T) {
	first := ""
	for i := 0; i < 25; i++ {
		recs, _ := FlattenJSONArray(strings.NewReader(sampleJSON), DefaultJSONCanon())
		got := DefaultJSONCanon().Render(recs)
		if i == 0 {
			first = got
			continue
		}
		if got != first {
			t.Fatal("Go map iteration is randomised; an unstable field order changes the token count between runs and makes any before/after comparison noise")
		}
	}
}

// The failure the flat format has and JSON does not.
func TestSeparatorInAValueIsEscapedNotSilentlyWritten(t *testing.T) {
	src := `[{"cmd": "grep a|b", "note": "x:y"}]`
	recs, err := FlattenJSONArray(strings.NewReader(src), DefaultJSONCanon())
	if err != nil {
		t.Fatal(err)
	}
	out := DefaultJSONCanon().Render(recs)
	// One record, two fields: exactly one unescaped field separator.
	unescaped := 0
	for i := 0; i < len(out); i++ {
		if out[i] == '|' && (i == 0 || out[i-1] != '\\') {
			unescaped++
		}
	}
	if unescaped != 1 {
		t.Fatalf("a raw separator in a value splits the record into the wrong number of fields "+
			"with no error raised, and the model reports on the mangled record confidently: %q", out)
	}
}

func TestNewlineInAValueCannotBreakTheRecordBoundary(t *testing.T) {
	src := `[{"a": "line1\nline2"}, {"b": "x"}]`
	recs, _ := FlattenJSONArray(strings.NewReader(src), DefaultJSONCanon())
	out := DefaultJSONCanon().Render(recs)
	if strings.Count(out, "\n") != 1 {
		t.Fatalf("two records means exactly one record separator; an embedded newline would forge a third record: %q", out)
	}
}

func TestNestedValuesAreRefusedNotStringified(t *testing.T) {
	src := `[{"a": {"deep": 1}}]`
	if _, err := FlattenJSONArray(strings.NewReader(src), DefaultJSONCanon()); err == nil {
		t.Fatal("flattening a nested value loses the nesting silently, and a consumer cannot tell it from a string containing delimiters")
	}
}

func TestLargeIntegerPrecisionSurvives(t *testing.T) {
	src := `[{"id": 9007199254740993}]`
	recs, err := FlattenJSONArray(strings.NewReader(src), DefaultJSONCanon())
	if err != nil {
		t.Fatal(err)
	}
	if recs[0].Fields[0].Value != "9007199254740993" {
		t.Fatalf("round-tripping through float64 loses precision on large ids: %q", recs[0].Fields[0].Value)
	}
}

func TestMalformedElementIsReportedNotSkipped(t *testing.T) {
	src := `[{"a": 1}, {"b": }]`
	if _, err := FlattenJSONArray(strings.NewReader(src), DefaultJSONCanon()); err == nil {
		t.Fatal("a silently dropped element is a region of the payload nothing downstream can discover was missing")
	}
}

func TestNonArrayInputIsRejected(t *testing.T) {
	if _, err := FlattenJSONArray(strings.NewReader(`{"a":1}`), DefaultJSONCanon()); err == nil {
		t.Fatal("a top-level object is not an array of records")
	}
}

func TestNullAndBoolRendering(t *testing.T) {
	recs, _ := FlattenJSONArray(strings.NewReader(`[{"a":null,"b":false,"c":true}]`), DefaultJSONCanon())
	got := DefaultJSONCanon().Render(recs)
	if got != "a:|b:0|c:1" {
		t.Fatalf("got %q", got)
	}
}

func TestSchemaHeaderNamesEveryFieldAndDecodesAbbreviations(t *testing.T) {
	recs, _ := FlattenJSONArray(strings.NewReader(sampleJSON), DefaultJSONCanon())
	c := DefaultJSONCanon()
	c.Abbrev = AutoAbbrev(recs, 4)
	hdr := c.SchemaHeader(recs)
	for _, k := range []string{"file_path", "symbol_type", "symbol_name", "is_exported"} {
		if !strings.Contains(hdr, k) {
			t.Fatalf("an abbreviation the header does not decode is unrecoverable; %q lacks %q", hdr, k)
		}
	}
}

func TestAutoAbbrevIsUnique(t *testing.T) {
	recs, _ := FlattenJSONArray(strings.NewReader(sampleJSON), DefaultJSONCanon())
	ab := AutoAbbrev(recs, 4)
	seen := map[string]string{}
	for full, short := range ab {
		if prev, dup := seen[short]; dup {
			t.Fatalf("%q and %q both abbreviate to %q — two fields would become indistinguishable", full, prev, short)
		}
		seen[short] = full
	}
}

func TestMeasureReportsCharsAndTokensSeparately(t *testing.T) {
	recs, _ := FlattenJSONArray(strings.NewReader(sampleJSON), DefaultJSONCanon())
	// A deliberately crude counter: whitespace-split. Stands in for a tokenizer only to prove
	// the two axes are reported independently.
	res := MeasureJSONCanon(sampleJSON, recs, func(s string) int { return len(strings.Fields(s)) })
	if res[0].Name != "original-json" {
		t.Fatal("the baseline must be the original payload — the question is cost against JSON, not against another flat form")
	}
	// Compaction must be measured as its own variant: it is free and lossless, and on a
	// pretty-printed payload it is the ENTIRE saving a flattened format credits to itself.
	var hasCompact bool
	for _, r := range res {
		if r.Name == "compacted-json" {
			hasCompact = true
		}
	}
	if !hasCompact {
		t.Fatal("without a compaction arm, flattening takes credit for removing whitespace")
	}
	for _, r := range res {
		if r.Chars == 0 || r.Tokens == 0 {
			t.Fatalf("%s reported nothing", r.Name)
		}
	}
	if res[1].Chars >= res[0].Chars {
		t.Fatal("the flat form is shorter in characters; that part is not in doubt")
	}
}
