/* Copyright 2026 the llama-herd authors
 * SPDX-License-Identifier: Apache-2.0
 *
 * lhchat implementation — see lhchat.h.
 *
 * Rendering and parsing both go through llama.cpp's common/chat.cpp, which is the only path
 * that understands tools. The core C API this project otherwise uses for templating
 * (`llama_chat_apply_template`) has no tools parameter at all, so an OpenAI `tools` array
 * reaching that call is discarded with nothing to report it.
 *
 * The parser is taken from the same apply() that produced the prompt. That coupling is
 * deliberate: the format a model emits tool calls in is a property of its template, so a
 * parser configured from anything else will read the right text with the wrong grammar and
 * report no calls rather than an error.
 */
#include "lhchat.h"

#include "chat.h"
#include "llama.h"

#include <cstring>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

/* Messages arrive in the same wire shape the plain template call already uses:
 * records joined by \x1e, each "role\x1fcontent". Kept identical so the two entry points
 * cannot disagree about what a message is. */
std::vector<common_chat_msg> parse_blob(const char *blob) {
    std::vector<common_chat_msg> msgs;
    std::string s(blob ? blob : "");
    size_t start = 0;
    while (start <= s.size()) {
        size_t rec = s.find('\x1e', start);
        std::string rec_s = (rec == std::string::npos) ? s.substr(start) : s.substr(start, rec - start);
        size_t unit = rec_s.find('\x1f');
        if (unit != std::string::npos) {
            common_chat_msg m;
            m.role = rec_s.substr(0, unit);
            m.content = rec_s.substr(unit + 1);
            msgs.push_back(std::move(m));
        }
        if (rec == std::string::npos) break;
        start = rec + 1;
    }
    return msgs;
}

/* OpenAI `tools` array -> common_chat_tool. Accepts both the wrapped form
 * ({"type":"function","function":{...}}) and a bare function object, because callers in this
 * codebase pass the wrapped one and hand-written probes often do not. */
std::vector<common_chat_tool> parse_tools(const char *tools_json) {
    std::vector<common_chat_tool> out;
    if (!tools_json || !*tools_json) return out;
    auto j = nlohmann::json::parse(tools_json, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_array()) return out;
    for (const auto &t : j) {
        const nlohmann::json *fn = &t;
        if (t.contains("function") && t["function"].is_object()) fn = &t["function"];
        if (!fn->contains("name")) continue;
        common_chat_tool tool;
        tool.name = fn->value("name", std::string());
        tool.description = fn->value("description", std::string());
        /* `parameters` is a JSON SCHEMA and common_chat_tool carries it as a STRING. Dumping
         * it (rather than passing the object) is what the upstream server does. */
        if (fn->contains("parameters")) tool.parameters = (*fn)["parameters"].dump();
        else tool.parameters = "{\"type\":\"object\",\"properties\":{}}";
        out.push_back(std::move(tool));
    }
    return out;
}

common_chat_tool_choice choice_of(int32_t c) {
    switch (c) {
        case 1:  return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        case 2:  return COMMON_CHAT_TOOL_CHOICE_NONE;
        default: return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }
}

/* Copy `s` into the caller's buffer. Returns the length written, or -needed when the buffer
 * is too small — the SAME convention the plain template call uses, so the Go binding's
 * grow-and-retry loop works unchanged for both. */
int32_t emit(const std::string &s, char *out, int32_t cap) {
    if (cap <= 0 || !out) return -(int32_t)(s.size() + 1);
    if ((int32_t)s.size() > cap) return -(int32_t)(s.size() + 1);
    std::memcpy(out, s.data(), s.size());
    return (int32_t)s.size();
}

/* Build the templates + inputs once; both entry points need the identical configuration, and
 * deriving the parser from a DIFFERENT apply() than the one that rendered the prompt is how a
 * parser silently stops matching the format it is parsing. */
bool build(llama_model *model, const char *blob, const char *tools_json, int32_t tool_choice,
           int add_ass, int enable_thinking, common_chat_templates_ptr &tmpls,
           common_chat_params &params) {
    tmpls = common_chat_templates_init(model, "");
    if (!tmpls) return false;
    common_chat_templates_inputs inputs;
    inputs.messages = parse_blob(blob);
    inputs.tools = parse_tools(tools_json);
    inputs.tool_choice = choice_of(tool_choice);
    inputs.add_generation_prompt = add_ass != 0;
    inputs.use_jinja = true;  /* tools are only honoured on the Jinja path */
    /* ⛔ MUST BE SET EXPLICITLY. common_chat_templates_inputs.enable_thinking defaults to TRUE
     * (common/chat.h) and v1 never touched it, so every tools render came out with thinking on.
     * Qwen3.x templates branch on exactly this field:
     *     {%- if enable_thinking is defined and enable_thinking is false %}
     *         {{- '<think>\n\n</think>\n\n' }}   {%- else %}   {{- '<think>\n' }}
     * so the default opens a reasoning block that a caller applying its own no-think prime then
     * nests a second one inside. */
    inputs.enable_thinking = enable_thinking != 0;
    if (inputs.messages.empty()) return false;
    params = common_chat_templates_apply(tmpls.get(), inputs);
    return true;
}

}  /* namespace */

extern "C" {

int lhchat_abi_version(void) { return LHCHAT_ABI_VERSION; }

/* v1 — RETAINED UNCHANGED IN BEHAVIOUR. Delegates with enable_thinking=1, which is precisely what
 * it did implicitly before v2 existed (the field defaults to true), so an older caller linked
 * against a newer lib sees no change. That is the whole reason v2 adds symbols beside these rather
 * than changing their signatures: the Go binding resolves by NAME at load time, so a changed
 * signature breaks callers silently at runtime instead of loudly at build time. */
int32_t lhchat_apply_template_tools(void *modelv, const char *blob, const char *tools_json,
                                         int32_t tool_choice, int add_ass, char *out, int32_t cap) {
    return lhchat_apply_template_tools_ex(modelv, blob, tools_json, tool_choice, add_ass,
                                          /*enable_thinking=*/1, out, cap);
}

int32_t lhchat_apply_template_tools_ex(void *modelv, const char *blob, const char *tools_json,
                                       int32_t tool_choice, int add_ass, int enable_thinking,
                                       char *out, int32_t cap) {
    if (!modelv || !blob) return -1;
    try {
        common_chat_templates_ptr tmpls;
        common_chat_params params;
        if (!build((llama_model *)modelv, blob, tools_json, tool_choice, add_ass, enable_thinking,
                   tmpls, params))
            return -2;
        return emit(params.prompt, out, cap);
    } catch (...) {
        /* ⛔ A template we cannot render is a REFUSAL, never a silently tool-less prompt.
         * Falling back to the no-tools path here would recreate this very bug: the caller
         * asked for tools, got a 200, and never learned they were dropped. */
        return -3;
    }
}

/* v1 — retained, delegates with enable_thinking=1 (its exact prior behaviour). */
int32_t lhchat_parse_output(void *modelv, const char *blob, const char *tools_json,
                                 int32_t tool_choice, const char *text, char *out, int32_t cap) {
    return lhchat_parse_output_ex(modelv, blob, tools_json, tool_choice, text,
                                  /*enable_thinking=*/1, out, cap);
}

int32_t lhchat_parse_output_ex(void *modelv, const char *blob, const char *tools_json,
                               int32_t tool_choice, const char *text, int enable_thinking,
                               char *out, int32_t cap) {
    if (!modelv || !text) return -1;
    try {
        common_chat_templates_ptr tmpls;
        common_chat_params params;
        /* Re-apply to recover the parser configuration that matches how the prompt was
         * rendered. Cheap relative to a decode, and stateless — the alternative is handing a
         * format id across the FFI boundary and trusting the two sides to agree about it. */
        if (!build((llama_model *)modelv, blob, tools_json, tool_choice, /*add_ass=*/1,
                   enable_thinking, tmpls, params))
            return -2;
        common_chat_parser_params pp(params);
        common_chat_msg msg = common_chat_parse(std::string(text), /*is_partial=*/false, pp);
        nlohmann::json j;
        j["content"] = msg.content;
        j["reasoning_content"] = msg.reasoning_content;
        nlohmann::json calls = nlohmann::json::array();
        for (const auto &c : msg.tool_calls) {
            calls.push_back({{"name", c.name}, {"arguments", c.arguments}, {"id", c.id}});
        }
        j["tool_calls"] = calls;
        return emit(j.dump(), out, cap);
    } catch (...) {
        return -3;
    }
}

}  /* extern "C" */
