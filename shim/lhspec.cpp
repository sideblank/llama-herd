// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

#include "lhspec.h"

#include "common.h"
#include "speculative.h"
#include "llama.h"

#include <cstring>
#include <string>
#include <vector>

namespace {

// holder keeps the C++ objects whose lifetime must match the driver's, and which cannot
// cross the C boundary.
// holder keeps the C++ objects whose lifetime must match the driver's and which cannot cross
// the C boundary.
//
// Construction is two-phase, which is not obvious from the header. init_from_params builds the
// draft side — for a prediction head that is a second context of MTP type over the same
// weights — and the caller must then place both contexts into the parameters before the
// driver itself is created. Creating the driver first yields one that has nothing to draft
// with.
struct holder {
    common_params                       params;
    common_speculative_init_result_ptr  init;
    common_speculative_ptr              spec;
    // prompts are retained because the draft parameters hold a pointer to one rather than
    // copying it, so each must outlive every draft call that references it.
    std::vector<llama_tokens>           prompts;
};

std::vector<std::string> split_csv(const char *s) {
    std::vector<std::string> out;
    if (!s || !*s) return out;
    std::string cur;
    for (const char *p = s; ; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
            if (*p == '\0') break;
        } else {
            cur.push_back(*p);
        }
    }
    return out;
}

} // namespace

extern "C" {

int lhspec_abi_version(void) { return LHSPEC_ABI_VERSION; }

int32_t lhspec_types_for_model(const char *gguf_path, char *buf, int32_t cap) {
    if (!gguf_path) return -1;
    std::string joined;
    try {
        const auto types = common_speculative_types_from_gguf(gguf_path);
        joined = common_speculative_type_name_str(types);
    } catch (...) {
        return -1;
    }
    const int32_t need = (int32_t) joined.size();
    if (buf && cap > 0) {
        const int32_t n = need < cap - 1 ? need : cap - 1;
        std::memcpy(buf, joined.data(), (size_t) n);
        buf[n] = '\0';
    }
    return need;
}

void *lhspec_init(void *model_tgt, void *ctx_tgt, const char *types,
                  int32_t n_seq, int32_t n_draft_max) {
    if (!model_tgt || !ctx_tgt) return nullptr;

    auto *h = new (std::nothrow) holder();
    if (!h) return nullptr;

    try {
        h->params.n_parallel = n_seq > 0 ? n_seq : 1;
        if (n_draft_max > 0) {
            h->params.speculative.draft.n_max = n_draft_max;
        }
        const auto names = split_csv(types);
        if (!names.empty()) {
            h->params.speculative.types = common_speculative_types_from_names(names);
        }

        // Phase one: build the draft side against the target.
        common_params params_dft = common_base_params_to_speculative(h->params);
        h->init = common_speculative_init_from_params(
            params_dft, (llama_model *) model_tgt, (llama_context *) ctx_tgt);
        if (!h->init || h->init->context() == nullptr) {
            delete h;
            return nullptr;
        }

        // Phase two: the driver needs both contexts, and only then can it be created.
        h->params.speculative.draft.ctx_tgt = (llama_context *) ctx_tgt;
        h->params.speculative.draft.ctx_dft = h->init->context();

        h->spec.reset(common_speculative_init(h->params.speculative,
                                              (uint32_t) h->params.n_parallel));
        if (!h->spec) {
            delete h;
            return nullptr;
        }
        h->prompts.resize((size_t) h->params.n_parallel);
    } catch (...) {
        delete h;
        return nullptr;
    }
    return h;
}

void lhspec_free(void *spec) {
    delete (holder *) spec;
}

void lhspec_begin(void *spec, int32_t seq_id, const int32_t *prompt, int32_t n_prompt) {
    auto *h = (holder *) spec;
    if (!h || seq_id < 0 || (size_t) seq_id >= h->prompts.size()) return;

    llama_tokens toks;
    toks.reserve((size_t) (n_prompt > 0 ? n_prompt : 0));
    for (int32_t i = 0; i < n_prompt; ++i) {
        toks.push_back((llama_token) prompt[i]);
    }
    h->prompts[(size_t) seq_id] = toks;
    common_speculative_begin(h->spec.get(), (llama_seq_id) seq_id, h->prompts[(size_t) seq_id]);
}

int32_t lhspec_process(void *spec, void *batch) {
    auto *h = (holder *) spec;
    if (!h || !batch) return -1;
    try {
        return common_speculative_process(h->spec.get(), *(const llama_batch *) batch) ? 0 : 1;
    } catch (...) {
        return -1;
    }
}

int32_t lhspec_draft(void *spec, int32_t seq_id, int32_t n_past, int32_t id_last,
                     int32_t n_max, int32_t *out, int32_t out_cap) {
    auto *h = (holder *) spec;
    if (!h || !out || out_cap <= 0) return -1;
    if (seq_id < 0 || (size_t) seq_id >= h->prompts.size()) return -1;

    try {
        auto &dp = common_speculative_get_draft_params(h->spec.get(), (llama_seq_id) seq_id);
        dp.drafting = true;
        dp.n_past   = (llama_pos) n_past;
        dp.id_last  = (llama_token) id_last;
        dp.n_max    = n_max;
        dp.prompt   = &h->prompts[(size_t) seq_id];

        common_speculative_draft(h->spec.get());

        // The driver clears drafting for every sequence at the end of the call, and leaves
        // the tokens it produced in result.
        const auto &dp2 = common_speculative_get_draft_params(h->spec.get(), (llama_seq_id) seq_id);
        if (!dp2.result) return 0;

        const int32_t n = (int32_t) dp2.result->size();
        const int32_t w = n < out_cap ? n : out_cap;
        for (int32_t i = 0; i < w; ++i) {
            out[i] = (int32_t) (*dp2.result)[(size_t) i];
        }
        return w;
    } catch (...) {
        return -1;
    }
}

void lhspec_accept(void *spec, int32_t seq_id, int32_t n_accepted) {
    auto *h = (holder *) spec;
    if (!h || n_accepted < 0) return;
    try {
        common_speculative_accept(h->spec.get(), (llama_seq_id) seq_id, (uint16_t) n_accepted);
    } catch (...) {
    }
}

} // extern "C"
