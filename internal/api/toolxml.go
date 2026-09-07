// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"encoding/json"
	"strings"
)

// Qwen-family models emit tool calls as XML that their own chat template specifies:
//
//	<tool_call>
//	<function=name>
//	<parameter=key>
//	value
//	</parameter>
//	</function>
//	</tool_call>
//
// llama.cpp has a parser for this, and on some revisions it does not match output that follows
// the template exactly. Its grammar permutes the REQUIRED parameters but requires optional ones
// to follow all of them, while the model emits parameters in schema-declaration order — so a
// call whose first parameter is optional parses as nothing. Measured against a live Qwen3.5
// model: a complete, correct call returned as prose, with no error anywhere.
//
// That failure is silent and indistinguishable from a model choosing not to call a tool, which
// makes it the worst kind: a caller that asked for tools gets a 200 and a plausible answer.
//
// So this reads the format the template documents. It runs ONLY after the library parser has
// been given the completion and returned nothing, and the response records which parser
// answered, so the two can never quietly disagree about a call.
//
// It does not validate against the tool schema. A model that invents a parameter should reach
// the caller as a visible bad call rather than vanish into a parse miss — that ambiguity is the
// thing this exists to remove.

// parseToolXML extracts tool calls from the template-mandated XML form, returning the calls and
// the remaining content. No calls means "no well-formed block here", which a caller must not
// read as "the model declined".
func parseToolXML(text string) (calls []respToolCall, content string) {
	var kept strings.Builder
	rest := text
	for {
		open := strings.Index(rest, "<tool_call>")
		if open < 0 {
			kept.WriteString(rest)
			break
		}
		closeAt := strings.Index(rest[open:], "</tool_call>")
		if closeAt < 0 {
			// Unterminated: keep it as text so a truncated generation stays visible instead of
			// being silently dropped.
			kept.WriteString(rest)
			break
		}
		end := open + closeAt + len("</tool_call>")
		if c, ok := parseFunctionBlock(rest[open+len("<tool_call>") : open+closeAt]); ok {
			kept.WriteString(rest[:open])
			calls = append(calls, c)
		} else {
			kept.WriteString(rest[:end])
		}
		rest = rest[end:]
	}
	return calls, strings.TrimSpace(kept.String())
}

// parseFunctionBlock reads `<function=NAME> <parameter=K>v</parameter>… </function>`.
func parseFunctionBlock(body string) (respToolCall, bool) {
	var out respToolCall
	fo := strings.Index(body, "<function=")
	if fo < 0 {
		return out, false
	}
	nameEnd := strings.Index(body[fo:], ">")
	if nameEnd < 0 {
		return out, false
	}
	name := strings.TrimSpace(body[fo+len("<function=") : fo+nameEnd])
	if name == "" {
		return out, false
	}
	args := map[string]any{}
	rest := body[fo+nameEnd:]
	for {
		po := strings.Index(rest, "<parameter=")
		if po < 0 {
			break
		}
		keyEnd := strings.Index(rest[po:], ">")
		if keyEnd < 0 {
			break
		}
		key := strings.TrimSpace(rest[po+len("<parameter=") : po+keyEnd])
		after := rest[po+keyEnd+1:]
		vEnd := strings.Index(after, "</parameter>")
		if vEnd < 0 {
			break
		}
		val := strings.Trim(after[:vEnd], "\n")
		if key != "" {
			// The template renders non-scalars as JSON, so recover those as structure. Anything
			// else stays a string — the prefix check keeps a bare word from becoming a number.
			var decoded any
			if (strings.HasPrefix(val, "{") || strings.HasPrefix(val, "[")) &&
				json.Unmarshal([]byte(val), &decoded) == nil {
				args[key] = decoded
			} else {
				args[key] = val
			}
		}
		rest = after[vEnd+len("</parameter>"):]
	}
	blob, err := json.Marshal(args)
	if err != nil {
		return out, false
	}
	out.Type = "function"
	out.Function = respToolFunction{Name: name, Arguments: string(blob)}
	return out, true
}
