// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0
//
// A build of the shim for llama.cpp revisions that predate the speculative API this project
// drives. It satisfies the same C ABI and reports that no speculation is available, so the
// binding links and the runtime declines drafting the same way it declines a quantization
// whose prediction head was stripped.
//
// It exists so the engine can be built and measured against an older library — comparing two
// revisions is the only way to tell a change in the library from a change in us, and being
// unable to build against the older one makes that comparison impossible.

#include "lhspec.h"

#include <cstring>

extern "C" {

int32_t lhspec_abi_version(void) { return LHSPEC_ABI_VERSION; }

int32_t lhspec_types_for_model(const char *, char *buf, int32_t cap) {
    if (buf && cap > 0) buf[0] = '\0';
    return 0;  // no speculative type is available on this build
}

void *lhspec_init(void *, void *, const char *, int32_t, int32_t,
                  int32_t, const char *, const char *, int32_t) {
    return nullptr;  // refused, which the caller reports and degrades from
}

void lhspec_free(void *) {}
void lhspec_begin(void *, int32_t, const int32_t *, int32_t) {}
int32_t lhspec_process(void *, void *) { return -1; }
int32_t lhspec_draft(void *, int32_t, int32_t, int32_t, int32_t, int32_t *, int32_t) { return -1; }
void lhspec_accept(void *, int32_t, int32_t) {}

}  // extern "C"
