// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
)

func parseOne(t *testing.T, body, marker string) (string, [][]byte, error) {
	t.Helper()
	var m message
	if err := json.Unmarshal([]byte(body), &m); err != nil {
		t.Fatal(err)
	}
	return m.parse(marker)
}

func TestPlainStringContent(t *testing.T) {
	text, media, err := parseOne(t, `{"role":"user","content":"hello"}`, "<img>")
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello" || len(media) != 0 {
		t.Fatalf("text=%q media=%d", text, len(media))
	}
}

func TestArrayContentWithImageInsertsMarker(t *testing.T) {
	img := base64.StdEncoding.EncodeToString([]byte("\xff\xd8\xff fake jpeg"))
	body := `{"role":"user","content":[
		{"type":"text","text":"what is "},
		{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,` + img + `"}},
		{"type":"text","text":" this?"}]}`

	text, media, err := parseOne(t, body, "<img>")
	if err != nil {
		t.Fatal(err)
	}
	if want := "what is <img> this?"; text != want {
		t.Fatalf("text = %q, want %q", text, want)
	}
	if len(media) != 1 {
		t.Fatalf("media = %d, want 1", len(media))
	}
	if !strings.HasPrefix(string(media[0]), "\xff\xd8\xff") {
		t.Fatalf("media bytes were not decoded correctly")
	}
}

// Fetching a caller-supplied URL server-side would let anyone reach whatever the server
// can, including cloud metadata and internal services.
func TestRemoteImageURLIsRefused(t *testing.T) {
	body := `{"role":"user","content":[
		{"type":"image_url","image_url":{"url":"http://169.254.169.254/latest/meta-data/"}}]}`
	_, _, err := parseOne(t, body, "<img>")
	if err == nil {
		t.Fatal("remote URLs must be refused")
	}
	if !strings.Contains(err.Error(), "inline data URLs") {
		t.Fatalf("error should explain why, got: %v", err)
	}
}

func TestMalformedDataURLsRejected(t *testing.T) {
	cases := map[string]string{
		"no comma":     `data:image/png;base64`,
		"not base64":   `data:image/png,rawbytes`,
		"bad encoding": `data:image/png;base64,!!!!not-base64!!!!`,
		"empty":        `data:image/png;base64,`,
	}
	for name, url := range cases {
		body := `{"role":"user","content":[{"type":"image_url","image_url":{"url":"` + url + `"}}]}`
		if _, _, err := parseOne(t, body, "<img>"); err == nil {
			t.Errorf("%s: should have been rejected", name)
		}
	}
}

func TestUnsupportedPartTypeRejected(t *testing.T) {
	body := `{"role":"user","content":[{"type":"video_url","text":"x"}]}`
	if _, _, err := parseOne(t, body, "<img>"); err == nil {
		t.Fatal("unknown part types should be rejected rather than silently dropped")
	}
}

func TestOversizeMediaRejected(t *testing.T) {
	big := base64.StdEncoding.EncodeToString(make([]byte, maxMediaBytes+1))
	body := `{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,` + big + `"}}]}`
	_, _, err := parseOne(t, body, "<img>")
	if err == nil || !strings.Contains(err.Error(), "over the") {
		t.Fatalf("oversize media should be rejected, got: %v", err)
	}
}
