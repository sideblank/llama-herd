/* Copyright 2026 the llama-herd authors
 * SPDX-License-Identifier: Apache-2.0
 *
 * lhchat stub — the same ABI as lhchat.cpp, reporting that tools cannot be rendered.
 *
 * The real shim needs llama.cpp's common library, which the image builds and a plain CI job
 * does not. Its purpose there is to check that the binding compiles and its tests pass, not
 * that a model renders tools, and without some lhchat the link fails on -llhchat — which
 * would be a finding about the workflow rather than about the code. Same reason lhspec_stub
 * exists, and it follows lhspec_stub's shape deliberately.
 *
 * Every entry point refuses. It does not fall back to rendering the messages without their
 * tools: a prompt missing its tools produces a confident answer to a request that was never
 * served, and a caller cannot tell that from a model that saw the tools and declined. That
 * silence is the failure the real shim exists to remove, so the stub must not reintroduce it
 * — a build linked against this reports that tools are unavailable, loudly, on first use.
 */
#include "lhchat.h"

extern "C" {

int lhchat_abi_version(void) { return LHCHAT_ABI_VERSION; }

int32_t lhchat_apply_template_tools(void *model, const char *blob, const char *tools_json,
                                    int32_t tool_choice, int add_ass, char *out, int32_t cap) {
    (void)model; (void)blob; (void)tools_json; (void)tool_choice;
    (void)add_ass; (void)out; (void)cap;
    return -3; /* render failed — the caller must surface this, never work around it */
}

int32_t lhchat_parse_output(void *model, const char *blob, const char *tools_json,
                            int32_t tool_choice, const char *text, char *out, int32_t cap) {
    (void)model; (void)blob; (void)tools_json; (void)tool_choice;
    (void)text; (void)out; (void)cap;
    return -3;
}

/* v2 (enable_thinking). Stubbed for the same reason and with the same refusal: a build that
 * links this reports that tools are unavailable rather than rendering them away silently. */
int32_t lhchat_apply_template_tools_ex(void *model, const char *blob, const char *tools_json,
                                       int32_t tool_choice, int add_ass, int enable_thinking,
                                       char *out, int32_t cap) {
    (void)model; (void)blob; (void)tools_json; (void)tool_choice;
    (void)add_ass; (void)enable_thinking; (void)out; (void)cap;
    return -3;
}

int32_t lhchat_parse_output_ex(void *model, const char *blob, const char *tools_json,
                               int32_t tool_choice, const char *text, int enable_thinking,
                               char *out, int32_t cap) {
    (void)model; (void)blob; (void)tools_json; (void)tool_choice;
    (void)text; (void)enable_thinking; (void)out; (void)cap;
    return -3;
}

}  /* extern "C" */
