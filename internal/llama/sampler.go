// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

/*
#include <stdlib.h>
#include "llama.h"
*/
import "C"

import "errors"

// SamplingParams configures one sampler chain.
//
// Zero values are not neutral for every field, so use DefaultSampling and override, rather
// than building this from scratch — a zero Temperature is greedy, not "unset".
type SamplingParams struct {
	// Temperature scales the logits. <= 0 selects greedy decoding and the other
	// randomness settings stop mattering.
	Temperature float32
	// TopK keeps the k most likely tokens. 0 disables.
	TopK int32
	// TopP keeps the smallest set whose probability mass reaches p. 1 disables.
	TopP float32
	// MinP keeps tokens at least p as likely as the most likely one. 0 disables.
	MinP float32
	// MinKeep floors how many candidates TopP and MinP may leave.
	MinKeep int

	// RepeatLastN is how many recent tokens the penalties consider. 0 disables.
	RepeatLastN int32
	// RepeatPenalty divides the logits of repeated tokens. 1 disables.
	RepeatPenalty float32
	// FreqPenalty and PresencePenalty are the OpenAI-style penalties. 0 disables.
	FreqPenalty     float32
	PresencePenalty float32

	// Seed makes sampling reproducible. 0 asks libllama for a random seed.
	Seed uint32
}

// DefaultSampling returns settings that behave like a conventional chat default.
func DefaultSampling() SamplingParams {
	return SamplingParams{
		Temperature:   0.8,
		TopK:          40,
		TopP:          0.95,
		MinP:          0.05,
		MinKeep:       1,
		RepeatLastN:   64,
		RepeatPenalty: 1.1,
	}
}

// Sampler is one sampler chain. It carries per-sequence state — the penalty window in
// particular — so every concurrent stream needs its own, and sharing one across streams
// would let their histories penalise each other.
type Sampler struct {
	c *C.struct_llama_sampler
	// nVocab bounds the logit scan taken by the greedy fast path.
	nVocab int32
	// greedy marks a chain that selects the most likely token with nothing modifying the
	// logits first, which can be evaluated without building a candidate array.
	greedy bool
}

// NewSampler builds a chain from params. nVocab comes from the model's vocabulary and is
// needed by the penalty sampler.
// greedyDirect reports whether a chain is exactly "take the most likely token" with nothing
// modifying the logits first. Such a chain can be evaluated by scanning the logits, which is
// what the fast path in Sample does.
func greedyDirect(p SamplingParams) bool {
	if p.Temperature > 0 {
		return false
	}
	// Penalties rewrite logits before the selection, so the scan would read the wrong values.
	return !(p.RepeatLastN > 0 && (p.RepeatPenalty != 1 || p.FreqPenalty != 0 || p.PresencePenalty != 0))
}

func NewSampler(p SamplingParams, nVocab int32) (*Sampler, error) {
	cp := C.llama_sampler_chain_default_params()
	chain := C.llama_sampler_chain_init(cp)
	if chain == nil {
		return nil, errors.New("llama: could not create sampler chain")
	}
	s := &Sampler{c: chain, nVocab: nVocab, greedy: greedyDirect(p)}

	// Order matters: penalties adjust logits, the truncations narrow the candidate set,
	// temperature scales what survives, and the final selector draws from it.
	if p.RepeatLastN > 0 && (p.RepeatPenalty != 1 || p.FreqPenalty != 0 || p.PresencePenalty != 0) {
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_penalties(
			C.int32_t(nVocab),
			C.int32_t(p.RepeatLastN),
			C.float(p.RepeatPenalty),
			C.float(p.FreqPenalty),
			C.float(p.PresencePenalty),
		))
	}

	if p.Temperature <= 0 {
		// Greedy: take the most likely token and ignore the rest of the settings.
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_greedy())
		return s, nil
	}

	minKeep := C.size_t(p.MinKeep)
	if p.MinKeep < 1 {
		minKeep = 1
	}
	if p.TopK > 0 {
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_k(C.int32_t(p.TopK)))
	}
	if p.TopP > 0 && p.TopP < 1 {
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_p(C.float(p.TopP), minKeep))
	}
	if p.MinP > 0 {
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_min_p(C.float(p.MinP), minKeep))
	}
	C.llama_sampler_chain_add(chain, C.llama_sampler_init_temp(C.float(p.Temperature)))
	C.llama_sampler_chain_add(chain, C.llama_sampler_init_dist(C.uint32_t(p.Seed)))

	return s, nil
}

// Sample draws the next token from the logits at batch index idx.
// Sample selects the next token for the batch entry at idx.
//
// A pure-greedy chain takes a direct scan of the logits instead of the library's general path.
// That path materialises one candidate record per vocabulary entry before selecting — 12 bytes
// times the vocabulary, per stream, per pass, written and then scanned — where the answer is a
// single pass over the logits already in memory. On a 150k vocabulary at four streams that is
// several megabytes of writes per forward pass, which competes for bandwidth the decode needs,
// and it matters most on the small hosts this runs on.
//
// The scan matches the library's greedy selection exactly, including its tie-break: start at
// the first entry and replace only on a strictly greater logit, so the earliest maximum wins in
// both. Anything else in the chain — penalties, temperature, truncation — falls through to the
// library, because those rewrite the logits the scan would read.
func (s *Sampler) Sample(ctx *Context, idx int32) Token {
	if s.greedy && s.nVocab > 0 {
		if logits := ctx.LogitsIth(idx, s.nVocab); len(logits) > 0 {
			best, bestV := 0, logits[0]
			for i := 1; i < len(logits); i++ {
				if logits[i] > bestV {
					bestV, best = logits[i], i
				}
			}
			return Token(best)
		}
	}
	return s.sampleViaChain(ctx, idx)
}

func (s *Sampler) sampleViaChain(ctx *Context, idx int32) Token {
	return Token(C.llama_sampler_sample(s.c, ctx.c, C.int32_t(idx)))
}

// Accept records a token as generated so the penalty window sees it. llama_sampler_sample
// already accepts the token it returns; call this only for tokens introduced another way,
// such as a prompt being replayed into the penalty history.
func (s *Sampler) Accept(t Token) { C.llama_sampler_accept(s.c, C.llama_token(t)) }

// Reset clears the chain's per-sequence state, which is what makes a slot safe to reuse for
// an unrelated request without its predecessor's history leaking in.
func (s *Sampler) Reset() { C.llama_sampler_reset(s.c) }

// Free releases the chain and every sampler added to it.
func (s *Sampler) Free() {
	if s == nil || s.c == nil {
		return
	}
	C.llama_sampler_free(s.c)
	s.c = nil
}
