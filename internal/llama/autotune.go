// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

import "runtime"

// AutoThreads picks a thread count for the machine this is running on.
//
// The library's default is four regardless of the host, which is wrong in both directions: it
// leaves a large machine idle during CPU inference, and it is more than a small one wants to
// give to work that is mostly waiting on a GPU.
//
// So it depends on where the weights are. With everything offloaded the CPU tokenizes, builds
// batches and reads back logits, and more threads buy nothing while competing with whatever
// else shares a rented node — a few are enough. Running on CPU, the forward pass itself is the
// work and wants every core.
//
// One core is left to the rest of the system in the CPU case. A machine saturated by its own
// inference schedules the process that has to read the results badly.
func AutoThreads(onGPU bool) int32 {
	n := runtime.NumCPU()
	if n < 1 {
		n = 1
	}
	if onGPU {
		// Enough to keep tokenization and batch assembly off the critical path without
		// contending for a small host's cores.
		if n > 4 {
			return 4
		}
		return int32(n)
	}
	if n > 1 {
		n--
	}
	return int32(n)
}

// AutoOutputsMax bounds the logit buffer by what the engine will actually request.
//
// Decoding asks for logits at one position per stream, or one per staged token when drafting.
// The library otherwise sizes that buffer for a whole prefill chunk — vocabulary times four
// bytes times n_batch — which on a 150k vocabulary with a 2048 batch is over a gigabyte of
// device memory reserved for positions nothing ever samples. On a card chosen because the
// weights barely fit, that is a gigabyte taken from the KV cache for nothing.
//
// It returns 0, meaning "leave it to the library", when prefill needs logits at every position:
// a drafter that predicts from hidden states reads them there, so the buffer really does have
// to hold a full chunk.
func AutoOutputsMax(streams, maxDraft int, outputEverywhere bool) uint32 {
	if outputEverywhere || streams < 1 {
		return 0
	}
	perStream := 1
	if maxDraft > 0 {
		perStream += maxDraft
	}
	n := streams * perStream
	// A small floor, so a single-stream model without drafting still has room for the
	// prefill position and a little slack rather than exactly one.
	if n < 8 {
		n = 8
	}
	return uint32(n)
}
