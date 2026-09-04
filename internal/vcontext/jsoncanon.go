// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// JSONRecord is one flattened object: ordered key/value pairs with the original element index kept.
//
// The index is kept because flattening is lossy. A downstream assertion naming a bad field has to
// point back at something, and once keys are abbreviated and structure is dropped there is nothing
// in the record itself that identifies the source element.
type JSONRecord struct {
	Index  int
	Fields []JSONField
}

// JSONField is one key/value pair after flattening.
type JSONField struct {
	Key   string
	Value string
}

// JSONCanon is a flattening configuration.
type JSONCanon struct {
	// FieldSep separates fields within a record; RecordSep separates records.
	FieldSep, RecordSep string
	// KeySep separates a key from its value.
	KeySep string
	// Abbrev maps full key names to short ones. Emitted in the schema header so the
	// abbreviation is recoverable rather than guessed.
	Abbrev map[string]string
}

// DefaultJSONCanon is the line-delimited form.
func DefaultJSONCanon() JSONCanon {
	return JSONCanon{FieldSep: "|", KeySep: ":", RecordSep: "\n", Abbrev: map[string]string{}}
}

// separatorSafe reports whether a value can be written without ambiguity.
//
// The delimiter must not occur in the data. A value containing the field separator produces a
// record that parses into the wrong number of fields with no error raised — the model reads a
// mangled record and reports on it confidently. This is the failure mode the format has and JSON
// does not, and it is the price of dropping the quoting.
func (c JSONCanon) separatorSafe(v string) bool {
	return !strings.Contains(v, c.FieldSep) &&
		!strings.Contains(v, c.RecordSep) &&
		!strings.Contains(v, c.KeySep)
}

// FlattenJSONArray converts a top-level JSON array of objects into line-delimited records.
//
// Streaming, via encoding/json's Decoder: a 256k-token payload decoded into map[string]any would
// allocate an interface box per value and a map per element, which is exactly the host-side garbage
// the fan-out cannot afford. The standard library streams natively — Token() to enter the array,
// then Decode into a json.RawMessage per element — so no third-party parser is required for this.
//
// Values that would collide with a separator are escaped rather than dropped or silently written.
func FlattenJSONArray(r io.Reader, c JSONCanon) ([]JSONRecord, error) {
	dec := json.NewDecoder(r)
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("vcontext: reading the opening token: %w", err)
	}
	if d, ok := tok.(json.Delim); !ok || d != '[' {
		return nil, fmt.Errorf("vcontext: expected a top-level array, found %v", tok)
	}

	var out []JSONRecord
	idx := 0
	for dec.More() {
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			// Reported, never skipped. A silently dropped element is a region of the payload
			// that was never judged and that nothing downstream can discover.
			return nil, fmt.Errorf("vcontext: element %d did not decode: %w", idx, err)
		}
		rec, err := flattenOne(raw, idx, c)
		if err != nil {
			return nil, err
		}
		out = append(out, rec)
		idx++
	}
	return out, nil
}

func flattenOne(raw json.RawMessage, idx int, c JSONCanon) (JSONRecord, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return JSONRecord{}, fmt.Errorf("vcontext: element %d is not an object: %w", idx, err)
	}
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	// Sorted, so the same input always produces the same records. Go map iteration is randomised,
	// and an unstable field order would change the token count between runs and make any
	// before/after comparison noise.
	sort.Strings(keys)

	rec := JSONRecord{Index: idx}
	for _, k := range keys {
		v, err := scalarString(obj[k])
		if err != nil {
			return JSONRecord{}, fmt.Errorf("vcontext: element %d field %q: %w", idx, k, err)
		}
		key := k
		if a, ok := c.Abbrev[k]; ok {
			key = a
		}
		if !c.separatorSafe(v) {
			v = escapeSeps(v, c)
		}
		rec.Fields = append(rec.Fields, JSONField{Key: key, Value: v})
	}
	return rec, nil
}

func escapeSeps(v string, c JSONCanon) string {
	rep := strings.NewReplacer(
		`\`, `\\`,
		c.FieldSep, `\`+c.FieldSep,
		c.KeySep, `\`+c.KeySep,
		c.RecordSep, `\n`,
	)
	return rep.Replace(v)
}

// scalarString renders a JSON value for the flat form.
//
// Nested objects and arrays are refused rather than stringified. Flattening a nested structure into
// a delimiter-separated line loses the nesting silently, and a consumer cannot tell a flattened
// sub-object from a string that happens to contain delimiters. A payload with real nesting needs
// sub-tree partitioning, not flattening.
func scalarString(raw json.RawMessage) (string, error) {
	t := strings.TrimSpace(string(raw))
	switch {
	case t == "null":
		return "", nil
	case t == "true":
		return "1", nil
	case t == "false":
		return "0", nil
	case strings.HasPrefix(t, "{"), strings.HasPrefix(t, "["):
		return "", fmt.Errorf("nested values cannot be flattened without losing structure; " +
			"partition the sub-tree instead")
	case strings.HasPrefix(t, `"`):
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", err
		}
		return s, nil
	default:
		// Number: preserved as written rather than round-tripped through float64, which would
		// lose precision on large integers and reformat exponents.
		if _, err := strconv.ParseFloat(t, 64); err != nil {
			return "", fmt.Errorf("unrecognised value %q", t)
		}
		return t, nil
	}
}

// Render writes the records in the flat form.
func (c JSONCanon) Render(recs []JSONRecord) string {
	var b strings.Builder
	for i, r := range recs {
		if i > 0 {
			b.WriteString(c.RecordSep)
		}
		for j, f := range r.Fields {
			if j > 0 {
				b.WriteString(c.FieldSep)
			}
			b.WriteString(f.Key)
			b.WriteString(c.KeySep)
			b.WriteString(f.Value)
		}
	}
	return b.String()
}

// SchemaHeader describes the flattening so a stream can read the records and a consumer can map an
// abbreviated key back to its original name.
//
// Small on purpose. It is prepended to every stream, so its cost is multiplied by the stream count
// — the same arithmetic that governs the skeleton and the symbol table.
func (c JSONCanon) SchemaHeader(recs []JSONRecord) string {
	seen := map[string]bool{}
	var keys []string
	for _, r := range recs {
		for _, f := range r.Fields {
			if !seen[f.Key] {
				seen[f.Key] = true
				keys = append(keys, f.Key)
			}
		}
	}
	rev := map[string]string{}
	for full, ab := range c.Abbrev {
		rev[ab] = full
	}
	var b strings.Builder
	fmt.Fprintf(&b, "records are %q-separated fields of key%svalue, one per line. fields: ",
		c.FieldSep, c.KeySep)
	for i, k := range keys {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(k)
		if full, ok := rev[k]; ok {
			fmt.Fprintf(&b, "=%s", full)
		}
	}
	return b.String()
}

// AutoAbbrev builds abbreviations for keys long enough to be worth shortening, keeping them unique.
//
// This is the only part of flattening that is MEASURED to save tokens. On 4,000 symbol records
// against the qwen3.5 vocabulary, abbreviating keys cut 21.0% of tokens while dropping the JSON
// syntax cut 0.0% — see docs/results/json-canonicalisation.md. Even here the character saving
// (58.7%) is 2.8x the token saving, because a BPE vocabulary already encodes a repeated key
// cheaply.
//
// It is not free: the schema header that decodes the abbreviations is prepended to every stream, so
// its cost is multiplied by the stream count. Measure both sides before enabling it.
func AutoAbbrev(recs []JSONRecord, minLen int) map[string]string {
	counts := map[string]int{}
	for _, r := range recs {
		for _, f := range r.Fields {
			counts[f.Key]++
		}
	}
	var keys []string
	for k := range counts {
		if len(k) >= minLen {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)

	out := map[string]string{}
	used := map[string]bool{}
	for _, k := range keys {
		for n := 1; n <= len(k); n++ {
			cand := k[:n]
			if !used[cand] {
				out[k] = cand
				used[cand] = true
				break
			}
		}
	}
	return out
}

// JSONCanonResult is one variant's measured cost.
type JSONCanonResult struct {
	Name   string
	Chars  int
	Tokens int
}

// MeasureJSONCanon counts each variant with a real tokenizer.
//
// The variants are measured against the ORIGINAL payload rather than against each other, because
// the question is not "is the flat form shorter" — it obviously is in characters — but "does it
// cost fewer tokens", and character count is not a proxy for token count.
func MeasureJSONCanon(raw string, recs []JSONRecord, count func(string) int) []JSONCanonResult {
	plain := DefaultJSONCanon()
	ab := plain
	ab.Abbrev = AutoAbbrev(recs, 4)

	abRecs := make([]JSONRecord, len(recs))
	for i, r := range recs {
		nr := JSONRecord{Index: r.Index}
		for _, f := range r.Fields {
			k := f.Key
			if a, ok := ab.Abbrev[k]; ok {
				k = a
			}
			nr.Fields = append(nr.Fields, JSONField{Key: k, Value: f.Value})
		}
		abRecs[i] = nr
	}

	flat := plain.Render(recs)
	flatAb := ab.Render(abRecs)

	out := []JSONCanonResult{{Name: "original-json", Chars: len(raw), Tokens: count(raw)}}
	// Compaction is measured as its own variant because it is almost always the whole win, and
	// it is the only one that is free: no format change, no escaping, no lost nesting, and it
	// round-trips. A flattening measured against PRETTY-PRINTED json credits itself with the
	// whitespace removal that this line does on its own.
	if c, err := CompactJSON(raw); err == nil {
		out = append(out, JSONCanonResult{Name: "compacted-json", Chars: len(c), Tokens: count(c)})
	}
	out = append(out,
		JSONCanonResult{Name: "flattened", Chars: len(flat), Tokens: count(flat)},
		JSONCanonResult{Name: "flattened+abbrev", Chars: len(flatAb), Tokens: count(flatAb)},
	)
	return out
}

// CompactJSON removes insignificant whitespace, changing nothing else.
//
// The cheapest and safest reduction available on a structured payload, and on a pretty-printed one
// it is measured to be the ENTIRE saving that a flattened format claims.
func CompactJSON(raw string) (string, error) {
	var buf bytes.Buffer
	if err := json.Compact(&buf, []byte(raw)); err != nil {
		return "", err
	}
	return buf.String(), nil
}
