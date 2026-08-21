/* lhspec — a pointer-only C ABI over llama.cpp's speculative decoding.
 *
 * Why a shim exists at all. Driving a multi-token-prediction head needs the target's hidden
 * states, and the function that returns them is not in the installed public header: it lives
 * in a staging header upstream describes as work in progress, permits breaking changes in,
 * and asks callers not to include. The supported way to reach that machinery is llama.cpp's
 * own common library, which upstream's server uses — but its interface is C++ and passes
 * std::vector and references, which cgo cannot bind.
 *
 * So this absorbs the C++ side and exposes pointers and primitives. It is deliberately thin:
 * it owns no policy, makes no decisions about when to draft, and holds no state beyond what
 * the underlying implementation already keeps. All three MTP architectures — a shared-memory
 * head, chained heads, and a single trained head — are handled by that implementation rather
 * than reimplemented here, which is the whole reason for going through it.
 */
#ifndef LHSPEC_H
#define LHSPEC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LHSPEC_ABI_VERSION 1
int lhspec_abi_version(void);

/* Names of the speculative types a model file supports, comma separated, written into buf.
 * Returns the number of bytes needed, which may exceed cap. A model with no speculative
 * capability yields an empty string — that is the cheap check for whether a quantization
 * kept its head, without loading the weights. */
int32_t lhspec_types_for_model(const char *gguf_path, char *buf, int32_t cap);

/* Create a speculative driver over an already-loaded target model and context.
 * `types` is a comma-separated list such as "draft-mtp"; empty selects whatever the model
 * supports. Returns NULL on failure.
 *
 * The draft context is a real context with its own KV cache, so the caller must pass the
 * target's geometry rather than letting it default. Defaulting is not a smaller version of
 * the right answer: an unset context size means "the length the model was trained with",
 * which for a long-context model is far larger than what the target was actually opened
 * with, and unset cache types mean f16 regardless of what the target quantized to. Both
 * together allocate enough to fail on a card the target fits comfortably — and the failure
 * arrives as a null context, indistinguishable from a model that has no head at all.
 *
 * n_ctx is the total across sequences, matching the target. type_k and type_v are ggml type
 * names such as "q8_0" or "f16"; NULL or empty means f16. flash_attn must match the target,
 * and is required by any quantized cache. */
void *lhspec_init(void *model_tgt, void *ctx_tgt, const char *types,
                  int32_t n_seq, int32_t n_draft_max,
                  int32_t n_ctx, const char *type_k, const char *type_v,
                  int32_t flash_attn);

void lhspec_free(void *spec);

/* Begin a generation for one sequence, giving the prompt it starts from. */
void lhspec_begin(void *spec, int32_t seq_id, const int32_t *prompt, int32_t n_prompt);

/* Show the driver the batch the target just decoded. This is where a head that predicts from
 * hidden states captures them, so it must be called after every target decode, not only when
 * a draft is wanted. Returns 0 on success. */
int32_t lhspec_process(void *spec, void *batch);

/* Ask for a draft for one sequence and copy it into out. Returns the number of tokens
 * written, 0 when the driver declined to draft, or negative on error. */
int32_t lhspec_draft(void *spec, int32_t seq_id, int32_t n_past, int32_t id_last,
                     int32_t n_max, int32_t *out, int32_t out_cap);

/* Report how many of the last draft's tokens the target kept. */
void lhspec_accept(void *spec, int32_t seq_id, int32_t n_accepted);

#ifdef __cplusplus
}
#endif
#endif /* LHSPEC_H */
