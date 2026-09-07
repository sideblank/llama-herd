/* Copyright 2026 the llama-herd authors
 * SPDX-License-Identifier: Apache-2.0
 *
 * lhchat — a pointer-only C ABI over llama.cpp's common/chat.cpp, so cgo can render and parse
 * TOOL CALLS.
 *
 * Why a shim, and why this one specifically: `llama_chat_apply_template` in the core C API
 * takes messages and nothing else. Tool definitions reach a model only through the Jinja path
 * in `common/chat.cpp`, whose interface passes std::vector and std::string by reference —
 * which cgo cannot bind, for the same reason lhspec exists. This absorbs the C++ side and
 * exposes JSON in, JSON out.
 *
 * Without it the server accepts an OpenAI `tools` array and silently discards it: the model is
 * never told the tools exist, answers from its own knowledge, and the reply is indistinguishable
 * from one where the model simply chose not to call anything. That failure is invisible to every
 * check except a human reading the answer, which is why it is worth a shim rather than a
 * best-effort prompt prefix.
 *
 * Rendering goes through the MODEL'S OWN template, so no per-model tool format is encoded here.
 * A model whose template formats tools differently is correct automatically; a hand-written
 * format table would be wrong for the first model that disagreed with it, silently.
 */
#ifndef LHCHAT_H
#define LHCHAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LHCHAT_ABI_VERSION 1

int lhchat_abi_version(void);

/* Render messages + tools into a prompt using the model's own chat template.
 *
 *   blob        messages as "role\x1fcontent" records joined by \x1e.
 *   tools_json  the OpenAI `tools` array, verbatim. NULL or empty renders without tools, so
 *               this is a safe superset of the plain template call.
 *   tool_choice 0 = auto, 1 = required, 2 = none.
 *
 * Returns bytes written, or -needed when `cap` is too small (grow and retry), or a negative
 * error: -1 bad arguments, -2 no template or no messages, -3 render failed.
 *
 * A -3 is a refusal and the caller must surface it. Falling back to a prompt without the tools
 * would return a plausible answer to a request that was not served, which is the failure this
 * exists to prevent.
 */
int32_t lhchat_apply_template_tools(void *model, const char *blob, const char *tools_json,
                                    int32_t tool_choice, int add_ass, char *out, int32_t cap);

/* Parse a completion into {"content","reasoning_content","tool_calls":[{name,arguments,id}]}.
 *
 * The blob, tools and choice must match the ones used to render: the parser is derived from the
 * render, and a parser derived from different inputs stops matching the format it is reading
 * without saying so.
 */
int32_t lhchat_parse_output(void *model, const char *blob, const char *tools_json,
                            int32_t tool_choice, const char *text, char *out, int32_t cap);

#ifdef __cplusplus
}
#endif

#endif /* LHCHAT_H */
