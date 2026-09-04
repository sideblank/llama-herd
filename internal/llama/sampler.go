// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

/*
#include <stdlib.h>
#include "llama.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"sync/atomic"
	"time"
	"unsafe"
)

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

	// Grammar constrains generation to a GBNF grammar, so output is structurally valid by
	// construction rather than by hope.
	//
	// This is what makes a deterministic merge possible: 48 streams each emitting a small,
	// schema-valid object can be combined by code, with no parsing that can fail on the 47th
	// response and no model asked to reconcile them. A grammar turns "usually parses" into
	// "cannot fail to parse", which is the difference between a pipeline and a demo.
	Grammar string
	// GrammarRoot is the grammar's start symbol. Defaults to "root".
	GrammarRoot string
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
	// grammar marks a chain that masks tokens to a grammar. Recorded so callers can tell a
	// constrained chain from an unconstrained one without inspecting the params again.
	grammar bool
	// truncN is how many candidates the chain is given, or 0 to give it the whole
	// vocabulary. See selectCandidates for why narrowing first is safe.
	truncN int
	// scratch holds the histogram used to pick a cutoff without sorting.
	hist [histBuckets]int32
	// cur is this chain's candidate buffer, allocated once and reused for every token.
	//
	// The library's one-call sampling helper builds a fresh vector of one record per
	// vocabulary entry on every token. At a 150k vocabulary that is a ~1.8 MB allocation per
	// token per stream, which is above the threshold where the allocator takes a fresh
	// mapping from the kernel — so each token pays a map, several hundred page faults, and an
	// unmap, none of which is sampling. Measured at 61% of engine time on a rented 3090, or
	// 4.3 ms per token, against 38% for the forward pass itself.
	//
	// Owned in C memory rather than Go's so the buffer is never moved or scanned by the
	// collector while a sampler chain is writing through it.
	cur *C.llama_token_data
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
	// A grammar masks tokens, so the highest logit may be one the grammar forbids. Scanning the
	// raw logits would pick it and produce output the grammar was supposed to make impossible.
	if p.Grammar != "" {
		return false
	}
	// Penalties rewrite logits before the selection, so the scan would read the wrong values.
	return !(p.RepeatLastN > 0 && (p.RepeatPenalty != 1 || p.FreqPenalty != 0 || p.PresencePenalty != 0))
}

// NewSampler builds a chain from params.
//
// vocab is required when p.Grammar is set — a grammar is compiled against the vocabulary it will
// constrain — and may be nil otherwise.
func NewSampler(p SamplingParams, nVocab int32, vocab *Vocab) (*Sampler, error) {
	cp := C.llama_sampler_chain_default_params()
	chain := C.llama_sampler_chain_init(cp)
	if chain == nil {
		return nil, errors.New("llama: could not create sampler chain")
	}
	s := &Sampler{c: chain, nVocab: nVocab, greedy: greedyDirect(p)}
	// Narrowing the candidate set before the chain sees it is only sound when the chain
	// itself truncates to a known count, so it is enabled by top-k and not otherwise.
	//
	// Why it is sound: every sampler that runs before the truncation can only LOWER a
	// logit — repeat penalty divides a positive logit and multiplies a negative one, and
	// the frequency and presence penalties subtract. So a token outside the kept set can
	// never climb into the final top-k. It can only fall further.
	//
	// Why the headroom is exactly the penalty window: at most RepeatLastN distinct tokens
	// are penalised at all, so keeping TopK+RepeatLastN candidates guarantees at least TopK
	// of them are untouched — and every untouched kept token outranks every token that was
	// dropped, by construction of the cutoff. The extra margin is slack, not correctness.
	if p.Temperature > 0 && p.TopK > 0 {
		n := int(p.TopK) + int(p.RepeatLastN) + 16
		if n < int(p.MinKeep) {
			n = int(p.MinKeep)
		}
		if int64(n) < int64(nVocab) {
			s.truncN = n
		}
	}
	// A greedy chain never builds candidates, so it does not need the buffer.
	if !s.greedy && nVocab > 0 {
		n := C.size_t(nVocab) * C.size_t(unsafe.Sizeof(C.llama_token_data{}))
		if buf := C.malloc(n); buf != nil {
			s.cur = (*C.llama_token_data)(buf)
		}
		// A nil buffer is not fatal: sampleViaChain falls back to the library's own helper,
		// which allocates per token but is correct.
	}

	// A grammar goes first, ahead of everything else.
	//
	// It masks tokens that cannot appear next, so every sampler after it is ranking only valid
	// continuations. Placed later it would be filtering a set the truncations had already
	// narrowed — and if top-k had dropped every grammatically valid token, there would be
	// nothing left to choose and generation would break in a way that looks like a model fault.
	if p.Grammar != "" {
		if vocab == nil {
			C.llama_sampler_free(chain)
			return nil, errors.New("llama: a grammar needs the vocabulary it constrains")
		}
		root := p.GrammarRoot
		if root == "" {
			root = "root"
		}
		cg := C.CString(p.Grammar)
		cr := C.CString(root)
		g := C.llama_sampler_init_grammar(vocab.c, cg, cr)
		C.free(unsafe.Pointer(cg))
		C.free(unsafe.Pointer(cr))
		if g == nil {
			C.llama_sampler_free(chain)
			return nil, fmt.Errorf("llama: grammar failed to parse (root %q)", root)
		}
		C.llama_sampler_chain_add(chain, g)
		s.grammar = true
	}

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

// sampleViaChain runs the full chain over a candidate buffer this sampler owns.
//
// It is the library's own sampling helper with one change: the candidate array is allocated
// once per chain instead of once per token. Everything else is kept deliberately identical,
// including the call to accept the chosen token — omitting that leaves the penalty samplers
// with an empty window, so repetition penalties quietly stop applying and the only symptom is
// prose that repeats itself.
func (s *Sampler) sampleViaChain(ctx *Context, idx int32) Token {
	if s.cur == nil || s.nVocab <= 0 {
		return Token(C.llama_sampler_sample(s.c, ctx.c, C.int32_t(idx)))
	}
	logits := ctx.LogitsIth(idx, s.nVocab)
	if len(logits) == 0 {
		return Token(C.llama_sampler_sample(s.c, ctx.c, C.int32_t(idx)))
	}

	cur := unsafe.Slice(s.cur, int(s.nVocab))
	selectStart := time.Now()
	n := s.selectCandidates(logits, cur)
	selectNanos.Add(time.Since(selectStart).Nanoseconds())
	if n == 0 {
		// Could not establish a cutoff. Hand over the whole vocabulary, which is what the
		// library would have done anyway.
		for i := range cur {
			cur[i].id = C.llama_token(i)
			cur[i].logit = C.float(logits[i])
			cur[i].p = 0
		}
		n = len(cur)
	}
	arr := C.llama_token_data_array{
		data:     s.cur,
		size:     C.size_t(n),
		selected: -1,
		sorted:   C.bool(false),
	}
	sampleCalls.Add(1)
	keptTotal.Add(int64(n))
	applyStart := time.Now()
	C.llama_sampler_apply(s.c, &arr)
	applyNanos.Add(time.Since(applyStart).Nanoseconds())
	if arr.selected < 0 || C.size_t(arr.selected) >= arr.size {
		// No sampler in the chain selected anything. Rather than invent a token, fall back
		// to the library helper so behaviour matches whatever it would have done.
		return Token(C.llama_sampler_sample(s.c, ctx.c, C.int32_t(idx)))
	}
	tok := unsafe.Slice(arr.data, int(arr.size))[arr.selected].id
	C.llama_sampler_accept(s.c, tok)
	return Token(tok)
}

// Sampler timings, accumulated across every chain in the process.
//
// They exist to answer one question that throughput cannot: of the time inside the sampler,
// how much is this package narrowing the candidate set, and how much is the library's chain
// running over what it was given. Optimising the wrong one of those has already happened once.
var (
	selectNanos atomic.Int64
	applyNanos  atomic.Int64
	sampleCalls atomic.Int64
	keptTotal   atomic.Int64
)

// SamplerTimings reports cumulative nanoseconds in candidate selection and in the library's
// chain, the number of chain samples taken, and the total candidates handed to the chain.
// keptTotal over calls is the average candidate-set size, which is how you tell truncation is
// actually happening rather than assumed to be.
func SamplerTimings() (selectNs, applyNs, calls, kept int64) {
	return selectNanos.Load(), applyNanos.Load(), sampleCalls.Load(), keptTotal.Load()
}

// histBuckets is the resolution of the cutoff search. More buckets means a tighter kept set
// and no change in correctness, since everything at or above the cutoff is kept.
const histBuckets = 1024

// histLo and histHi bound the logit range the histogram covers. Values above the top are
// clamped into the last bucket and values below the bottom are ignored — neither can be in the
// top few hundred of a vocabulary, and if that assumption ever fails the cutoff search simply
// does not find enough candidates and the whole vocabulary is used instead.
const (
	histLo = -60.0
	histHi = 60.0
)

// histStride is how many logits the cutoff estimate skips. The collect pass still reads every
// logit; this only thins the pass that decides where to cut.
const histStride = 8

// histMargin is how many buckets below the estimated cutoff to actually cut, absorbing the
// error introduced by sampling. Keeping extra candidates is close to free.
const histMargin = 8

// selectCandidates fills dst with every token whose logit is at or above a cutoff chosen so at
// least truncN survive, and returns how many. Returns 0 when truncation is disabled or no
// usable cutoff exists, meaning the caller should fall back to the whole vocabulary.
//
// Two linear passes and no sort. The alternative — handing the chain all 151,936 entries —
// costs a hash lookup per entry in the penalty sampler and a partial sort over the same range
// in top-k. Narrowing first made the library's chain essentially free: measured at 0.014 ms per
// token against 0.98 ms for this selection, which is why the selection itself is written to
// touch the logits as few times as possible.
//
// A fixed bucket range rather than one anchored on the maximum is what removes the third pass:
// finding the maximum first would mean reading every logit an extra time purely to decide where
// the buckets start, and the walk down from the top bucket finds the cutoff without it.
//
// Everything at or above the cutoff is kept, ties included, so the cut never splits a group of
// equal logits.
func (s *Sampler) selectCandidates(logits []float32, dst []C.llama_token_data) int {
	if s.truncN <= 0 || len(logits) == 0 || s.truncN >= len(logits) {
		return 0
	}

	cutoff, ok := s.cutoffFor(logits, histStride)
	if !ok {
		// The estimate could not place a cutoff. Try again reading every logit before
		// giving up, since a sparse distribution is exactly when the sample is unreliable.
		if cutoff, ok = s.cutoffFor(logits, 1); !ok {
			return 0
		}
	}

	n := 0
	for i, v := range logits {
		if v >= cutoff {
			dst[n].id = C.llama_token(i)
			dst[n].logit = C.float(v)
			dst[n].p = 0
			n++
		}
	}
	if n < s.truncN {
		// The sampled cutoff was too high. Fall back to reading every logit; if that still
		// cannot find enough, the caller uses the whole vocabulary.
		cutoff, ok = s.cutoffFor(logits, 1)
		if !ok {
			return 0
		}
		n = 0
		for i, v := range logits {
			if v >= cutoff {
				dst[n].id = C.llama_token(i)
				dst[n].logit = C.float(v)
				dst[n].p = 0
				n++
			}
		}
		if n < s.truncN {
			return 0
		}
	}
	return n
}

// cutoffFor estimates a logit cutoff that leaves at least truncN candidates above it, reading
// every stride'th logit.
//
// Sampling is sound here because the cutoff only has to be LOW enough. Reading one logit in
// eight gives an eighth of the counts, so the target count is scaled to match, and the result
// is then stepped down by a margin — an estimate that errs low keeps more candidates than
// necessary, which costs nothing, since handing the chain a few hundred entries instead of a
// hundred was measured at 0.01 ms. An estimate that errs high is caught by the caller, which
// recomputes over every logit.
//
// This is what brings selection close to its floor: the collect pass has to read every logit,
// but nothing else does.
func (s *Sampler) cutoffFor(logits []float32, stride int) (float32, bool) {
	hist := &s.hist
	for i := range hist {
		hist[i] = 0
	}
	const scale = float32(histBuckets) / float32(histHi-histLo)
	counted := 0
	for i := 0; i < len(logits); i += stride {
		b := int32((logits[i] - float32(histLo)) * scale)
		if b < 0 {
			continue // too far down to matter, and NaN lands here too
		}
		if b >= histBuckets {
			b = histBuckets - 1
		}
		hist[b]++
		counted++
	}

	// Scale the target to the sample, never below one bucket's worth.
	target := int32(s.truncN / stride)
	if target < 1 {
		target = 1
	}
	cum, bucket := int32(0), -1
	for i := histBuckets - 1; i >= 0; i-- {
		cum += hist[i]
		if cum >= target {
			bucket = i
			break
		}
	}
	if bucket < 0 {
		return 0, false
	}
	// Step down for margin: a sampled estimate is noisy, and being too low is free.
	bucket -= histMargin
	if bucket < 0 {
		bucket = 0
	}
	return float32(histLo) + float32(bucket)/scale, true
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
	if s.cur != nil {
		C.free(unsafe.Pointer(s.cur))
		s.cur = nil
	}
}

// GrammarConstrained reports whether this chain masks tokens to a grammar.
//
// Exposed because the fast paths must not be taken when it is set: scanning raw logits would
// select the highest-scoring token regardless of whether the grammar permits it, producing exactly
// the output the grammar existed to make impossible.
func (s *Sampler) GrammarConstrained() bool { return s != nil && s.grammar }
