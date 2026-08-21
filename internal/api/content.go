// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

// message is a request message before it is reduced to text plus media.
//
// Content is either a plain string or the array-of-parts form clients use for images.
// Both shapes are in active use, so both are accepted.
type message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// contentPart is one element of the array form.
type contentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL *struct {
		URL string `json:"url"`
	} `json:"image_url,omitempty"`
}

// maxMediaBytes bounds a single decoded image.
const maxMediaBytes = 32 << 20

// parse reduces a message to prompt text and any media it carried. Each image is replaced
// in the text by marker, which is where the model expects the encoded media to sit.
func (m message) parse(marker string) (string, [][]byte, error) {
	if len(m.Content) == 0 || string(m.Content) == "null" {
		return "", nil, nil
	}

	// The common case: content is a plain string.
	if m.Content[0] == '"' {
		var s string
		if err := json.Unmarshal(m.Content, &s); err != nil {
			return "", nil, err
		}
		return s, nil, nil
	}

	var parts []contentPart
	if err := json.Unmarshal(m.Content, &parts); err != nil {
		return "", nil, fmt.Errorf("content must be a string or an array of parts: %w", err)
	}

	var sb strings.Builder
	var media [][]byte
	for i, p := range parts {
		switch p.Type {
		case "text", "input_text":
			sb.WriteString(p.Text)
		case "image_url", "input_image":
			if p.ImageURL == nil || p.ImageURL.URL == "" {
				return "", nil, fmt.Errorf("part %d is an image with no url", i)
			}
			raw, err := decodeDataURL(p.ImageURL.URL)
			if err != nil {
				return "", nil, fmt.Errorf("part %d: %w", i, err)
			}
			media = append(media, raw)
			// The marker is what tells the tokenizer where the encoded image goes. A
			// prompt without it drops the image silently.
			sb.WriteString(marker)
		default:
			return "", nil, fmt.Errorf("part %d has unsupported type %q", i, p.Type)
		}
	}
	return sb.String(), media, nil
}

// decodeDataURL accepts only inline data URLs.
//
// Remote URLs are refused deliberately. Fetching a caller-supplied URL server-side turns
// this endpoint into a request forwarder that can reach anything the server can — cloud
// metadata endpoints, internal services, private networks — on behalf of whoever sends the
// request. Clients that want to send an image can inline it.
func decodeDataURL(u string) ([]byte, error) {
	if !strings.HasPrefix(u, "data:") {
		return nil, fmt.Errorf("only inline data URLs are accepted; " +
			"fetching remote URLs server-side would let a caller reach any host this server can")
	}
	comma := strings.IndexByte(u, ',')
	if comma < 0 {
		return nil, fmt.Errorf("malformed data URL")
	}
	meta, payload := u[5:comma], u[comma+1:]
	if !strings.Contains(meta, "base64") {
		return nil, fmt.Errorf("data URL must be base64 encoded")
	}
	raw, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return nil, fmt.Errorf("data URL is not valid base64: %w", err)
	}
	if len(raw) == 0 {
		return nil, fmt.Errorf("data URL decoded to nothing")
	}
	if len(raw) > maxMediaBytes {
		return nil, fmt.Errorf("media is %d bytes, over the %d limit", len(raw), maxMediaBytes)
	}
	return raw, nil
}
