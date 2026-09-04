// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/sideblank/llama-herd/internal/llama"
)

// latentProbe answers the one question the whole of HLSR rests on: does a chunk's final hidden
// state carry enough of that chunk to seed a grounded generation?
//
// It is deliberately a small, cheap command rather than part of the pipeline. The premise is
// falsifiable in seconds on a 0.5B, and if it is false nothing downstream — pooling, projection,
// dispatch — is worth writing. Running it first is the difference between an afternoon and a week.
//
// The sharp test is separation, not recall. Two chunks with conflicting specifics injected as a
// two-vector sequence: if the answer recovers both values AND attributes them correctly, positional
// separation is doing real work. Mixing them up is the failure a pooled design produces by
// construction, so seeing it here would say the representation carries topic but not detail.
func latentProbe(args []string) int {
	fs := newFlagSet("latent-probe")
	modelPath := fs.String("model", "", "path to a small GGUF (required)")
	instruction := fs.String("instruction",
		"Synthesize the primary key entities, facts, and assertions above into a dense representation:",
		"trailing text that turns the final position into a summary-predicting state")
	control := fs.Bool("control", false, "omit the instruction, to show what raw h_last gives")
	asTokens := fs.Bool("as-tokens", false,
		"inject each chunk's top predicted TOKEN as text instead of its hidden vector, to measure "+
			"how much the vector carries beyond that token")
	scales := fs.String("scale", "1.0", "comma-separated scale factors to try on the injected vectors")
	tokens := fs.Int("tokens", 60, "tokens to generate from the injected vectors")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "latent-probe: --model is required")
		return 2
	}

	chunks := []string{
		"System A is the primary ingest node. System A uses port 8080 for all inbound traffic.",
		"System B is the reporting node. System B uses port 9090 for all inbound traffic.",
	}
	question := "\n\nQuestion: What ports do System A and System B use? Answer with both.\n\nAnswer:"

	llama.Backend()
	defer llama.BackendFree()

	mp := llama.DefaultModelParams()
	m, err := llama.LoadModel(*modelPath, mp)
	if err != nil {
		fmt.Fprintf(os.Stderr, "latent-probe: %v\n", err)
		return 1
	}
	defer m.Free()

	nEmbd := m.NEmbd()
	vocab := m.Vocab()

	// Embeddings must be on, or llama_get_embeddings_ith returns nothing and the probe reports
	// a failure that is really a configuration mistake.
	cp := llama.DefaultContextParams()
	cp.NCtx = 4096
	cp.NSeqMax = 4
	cp.Embeddings = true
	ctx, err := llama.NewContext(m, cp)
	if err != nil {
		fmt.Fprintf(os.Stderr, "latent-probe: %v\n", err)
		return 1
	}
	defer ctx.Free()

	fmt.Printf("model: %s\n", m.Summary())
	fmt.Printf("n_embd: %d   instruction: %v\n\n", nEmbd, !*control)

	// --- extract one hidden state per chunk -----------------------------------------------
	batch := llama.NewBatch(2048, 4)
	defer batch.Free()

	states := make([][]float32, len(chunks))
	direct := make([]string, len(chunks))
	for i, chunk := range chunks {
		text := chunk
		if !*control {
			text = chunk + "\n---\n" + *instruction
		}
		toks, err := vocab.Tokenize(text, true, true)
		if err != nil {
			fmt.Fprintf(os.Stderr, "tokenize: %v\n", err)
			return 1
		}
		seq := llama.SeqID(i)
		ctx.SeqRm(seq, -1, -1)
		batch.Clear()
		for j, t := range toks {
			last := j == len(toks)-1
			if err := batch.Add(t, llama.Pos(j), []llama.SeqID{seq}, last); err != nil {
				fmt.Fprintf(os.Stderr, "stage: %v\n", err)
				return 1
			}
		}
		if err := ctx.Decode(batch); err != nil {
			fmt.Fprintf(os.Stderr, "decode chunk %d: %v\n", i, err)
			return 1
		}
		ctx.Synchronize()

		// -1 is the last output position, which is where the instruction lands.
		h := ctx.EmbeddingsIth(-1, nEmbd)
		if len(h) == 0 {
			fmt.Fprintln(os.Stderr, "latent-probe: no embeddings returned — is the context "+
				"built with Embeddings enabled and the last token marked for output?")
			return 1
		}
		states[i] = append([]float32(nil), h...) // copy: the library reuses its buffer

		// What the model itself would say from this position, before any injection.
		//
		// This separates two failures that look identical downstream: the detail was never in
		// the state, or it was there and did not survive injection. They have entirely
		// different fixes — a better instruction versus a better projection — and without this
		// the probe cannot tell which one it is measuring.
		direct[i] = continueFrom(ctx, vocab, batch, seq, len(toks), 12)
		fmt.Printf("chunk %d: %d tokens -> h_last[%d]  L2=%.1f\n", i, len(toks), len(h), l2(h))
		fmt.Printf("          model continues directly with: %q\n", direct[i])
	}

	// --- inject at several scales -----------------------------------------------------------
	//
	// A final hidden state and an input embedding differ in magnitude as well as in space. If the
	// injected vector is far out of the input distribution by norm alone, generation degenerates
	// for a reason that has nothing to do with whether the vector carries chunk content — so scale
	// is swept before the premise is judged. A scale that produces coherent output says the
	// mismatch is fixable; none working says the direction is wrong too.
	if *asTokens {
		// The comparison that bounds what a soft-token is worth.
		//
		// For a tied-embedding model the natural h_out -> h_in map is the expected input
		// embedding under the predicted next-token distribution — which means an injected vector
		// carries approximately ONE TOKEN of information, plus the shape of the distribution
		// around it. If injecting the vector does no better than appending its top token as
		// text, that is the whole story, and 48 soft-tokens are worth about a 48-token summary
		// rather than a compression of 256k.
		return runAsTokens(ctx, vocab, direct, question, *tokens)
	}

	for _, sc := range parseScales(*scales) {
		if code := runAt(ctx, m, vocab, states, question, sc, *tokens, nEmbd); code != 0 {
			return code
		}
	}
	return 0
}

// runAsTokens builds the answer context from each chunk's own top continuation, as ordinary text.
func runAsTokens(ctx *llama.Context, vocab *llama.Vocab, direct []string, question string, tokens int) int {
	answer := llama.SeqID(3)
	ctx.SeqRm(answer, -1, -1)

	batch := llama.NewBatch(2048, 4)
	defer batch.Free()

	var b strings.Builder
	for i, d := range direct {
		fmt.Fprintf(&b, "Chunk %d: %s\n", i, firstToken(d))
	}
	b.WriteString(question)
	prompt := b.String()
	fmt.Printf("\n--- as text, one token per chunk ---\ncontext given:\n%s\n", prompt)

	toks, err := vocab.Tokenize(prompt, true, true)
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenize: %v\n", err)
		return 1
	}
	batch.Clear()
	for j, t := range toks {
		if err := batch.Add(t, llama.Pos(j), []llama.SeqID{answer}, j == len(toks)-1); err != nil {
			fmt.Fprintf(os.Stderr, "stage: %v\n", err)
			return 1
		}
	}
	if err := ctx.Decode(batch); err != nil {
		fmt.Fprintf(os.Stderr, "decode: %v\n", err)
		return 1
	}

	text := continueFrom(ctx, vocab, batch, answer, len(toks), tokens)
	fmt.Printf("\ngenerated:\n%s\n\n", text)
	fmt.Printf("  8080: %v   9090: %v\n", strings.Contains(text, "8080"), strings.Contains(text, "9090"))
	return 0
}

// firstToken keeps only the leading word of a continuation, approximating one token.
func firstToken(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexAny(s, " \n\t"); i > 0 {
		return s[:i]
	}
	return s
}

// continueFrom generates a few tokens from wherever a sequence currently ends, without disturbing
// it — the state is trimmed back afterwards so the caller's cache is unchanged.
func continueFrom(ctx *llama.Context, vocab *llama.Vocab, batch *llama.Batch,
	seq llama.SeqID, pos int, n int) string {

	sampler, err := llama.NewSampler(llama.SamplingParams{Temperature: 0}, vocab.NTokens(), vocab)
	if err != nil {
		return ""
	}
	defer sampler.Free()

	var out strings.Builder
	p := pos
	for i := 0; i < n; i++ {
		tok := sampler.Sample(ctx, -1)
		if tok == vocab.EOS() {
			break
		}
		piece, err := vocab.TokenToPiece(llama.Token(tok), false)
		if err != nil {
			break
		}
		out.Write(piece)
		batch.Clear()
		if err := batch.Add(llama.Token(tok), llama.Pos(p), []llama.SeqID{seq}, true); err != nil {
			break
		}
		if err := ctx.Decode(batch); err != nil {
			break
		}
		p++
	}
	// Leave the sequence as it was found.
	ctx.SeqRm(seq, llama.Pos(pos), -1)
	return strings.TrimSpace(out.String())
}

func l2(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return mathSqrt(s)
}

func mathSqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 40; i++ {
		z = 0.5 * (z + x/z)
	}
	return z
}

func parseScales(s string) []float32 {
	var out []float32
	for _, f := range strings.Split(s, ",") {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		var v float64
		if _, err := fmt.Sscanf(f, "%g", &v); err == nil && v != 0 {
			out = append(out, float32(v))
		}
	}
	if len(out) == 0 {
		out = []float32{1}
	}
	return out
}

func runAt(ctx *llama.Context, m *llama.Model, vocab *llama.Vocab, states [][]float32,
	question string, scale float32, tokens int, nEmbd int32) int {

	answer := llama.SeqID(3)
	ctx.SeqRm(answer, -1, -1)

	eb, err := llama.NewEmbdBatch(int32(len(states)), nEmbd, 4)
	if err != nil {
		fmt.Fprintf(os.Stderr, "latent-probe: %v\n", err)
		return 1
	}
	defer eb.Free()
	for i, h := range states {
		// No learned projection here on purpose. If plain injection already grounds the answer,
		// W_proj (#21) is unnecessary; if only a scaled version works, the projection is mostly a
		// magnitude fix; if nothing works, it needs to be learned or the premise is false.
		v := make([]float32, len(h))
		for j, x := range h {
			v[j] = x * scale
		}
		if err := eb.Add(v, llama.Pos(i), []llama.SeqID{answer}, false); err != nil {
			fmt.Fprintf(os.Stderr, "stage vector %d: %v\n", i, err)
			return 1
		}
	}
	if err := ctx.DecodeEmbd(eb); err != nil {
		fmt.Fprintf(os.Stderr, "latent-probe: injecting vectors failed: %v\n", err)
		return 1
	}

	batch := llama.NewBatch(2048, 4)
	defer batch.Free()

	qToks, err := vocab.Tokenize(question, false, true)
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenize question: %v\n", err)
		return 1
	}
	batch.Clear()
	pos := len(states)
	for j, t := range qToks {
		last := j == len(qToks)-1
		if err := batch.Add(t, llama.Pos(pos+j), []llama.SeqID{answer}, last); err != nil {
			fmt.Fprintf(os.Stderr, "stage question: %v\n", err)
			return 1
		}
	}
	if err := ctx.Decode(batch); err != nil {
		fmt.Fprintf(os.Stderr, "decode question: %v\n", err)
		return 1
	}
	pos += len(qToks)

	// --- greedy generation ----------------------------------------------------------------
	sampler, err := llama.NewSampler(llama.SamplingParams{Temperature: 0}, vocab.NTokens(), vocab)
	if err != nil {
		fmt.Fprintf(os.Stderr, "sampler: %v\n", err)
		return 1
	}
	defer sampler.Free()

	var out strings.Builder
	for i := 0; i < tokens; i++ {
		tok := sampler.Sample(ctx, -1)
		if tok == vocab.EOS() {
			break
		}
		piece, err := vocab.TokenToPiece(llama.Token(tok), false)
		if err != nil {
			break
		}
		out.Write(piece)
		batch.Clear()
		if err := batch.Add(llama.Token(tok), llama.Pos(pos), []llama.SeqID{answer}, true); err != nil {
			break
		}
		if err := ctx.Decode(batch); err != nil {
			break
		}
		pos++
	}

	text := out.String()
	fmt.Printf("\n--- scale %.4g ---\n%s\n", scale, strings.TrimSpace(text))

	// --- verdict --------------------------------------------------------------------------
	has8080 := strings.Contains(text, "8080")
	has9090 := strings.Contains(text, "9090")
	fmt.Printf("  8080: %v   9090: %v\n", has8080, has9090)
	_ = nEmbd
	switch {
	case has8080 && has9090:
		fmt.Println("PASS — both values recovered from injected state. Check attribution above: " +
			"A→8080 and B→9090 means positional separation is carrying detail.")
	case has8080 || has9090:
		fmt.Println("PARTIAL — one value recovered. The representation carries some detail but " +
			"not both; this bounds what HLSR can be used for.")
	default:
		fmt.Println("FAIL — neither value recovered. If the control also fails, the premise is " +
			"false and the pooling and projection work should not be started.")
	}
	return 0
}
