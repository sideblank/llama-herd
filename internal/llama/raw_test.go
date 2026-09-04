package llama

import (
	"os"
	"strings"
	"testing"
)

// TestRawGeneration drives the binding directly: no engine, no chat template, greedy
// sampling. It separates "this model produces sensible logits under this build" from every
// layer above it, which is the first question worth answering when output looks wrong.
//
// Set LH_MODEL to a GGUF to run it, and LH_BACKENDS to the directory holding the ggml
// backend plugins if they do not sit beside the library. It skips otherwise, so it costs
// nothing in CI and is there when a model is misbehaving.
func TestRawGeneration(t *testing.T) {
	path := os.Getenv("LH_MODEL")
	if path == "" {
		t.Skip("LH_MODEL not set")
	}
	Backend()
	defer BackendFree()
	if dir := os.Getenv("LH_BACKENDS"); dir != "" {
		LoadBackendsFrom(dir)
	} else {
		LoadBackends()
	}

	mp := DefaultModelParams()
	mp.NGPULayers = 0
	m, err := LoadModel(path, mp)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()

	cp := DefaultContextParams()
	cp.NCtx = 2048
	cp.NBatch = 512
	cp.NSeqMax = 1
	cp.NThreads = 24
	cp.NThreadsBatch = 24
	ctx, err := NewContext(m, cp)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Free()

	vocab := m.Vocab()
	toks, err := vocab.Tokenize("The sea is", true, true)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("prompt tokens: %v", toks)

	b := NewBatch(512, 1)
	defer b.Free()
	for i, tk := range toks {
		if err := b.Add(tk, Pos(i), []SeqID{0}, i == len(toks)-1); err != nil {
			t.Fatal(err)
		}
	}
	if err := ctx.Decode(b); err != nil {
		t.Fatalf("prefill decode: %v", err)
	}

	smpl, err := NewSampler(SamplingParams{}, vocab.NTokens(), vocab)
	if err != nil {
		t.Fatal(err)
	}
	defer smpl.Free()

	var out strings.Builder
	pos := Pos(len(toks))
	for i := 0; i < 24; i++ {
		tk := smpl.Sample(ctx, -1)
		if tk == vocab.EOS() {
			t.Logf("EOS at step %d (token %d)", i, tk)
			break
		}
		piece, _ := vocab.TokenToPiece(tk, false)
		out.Write(piece)
		smpl.Accept(tk)

		b.Clear()
		if err := b.Add(tk, pos, []SeqID{0}, true); err != nil {
			t.Fatal(err)
		}
		if err := ctx.Decode(b); err != nil {
			t.Fatalf("decode step %d: %v", i, err)
		}
		pos++
	}
	t.Logf("GENERATED>>>%s<<<", out.String())
	if out.Len() == 0 {
		t.Fatal("model produced nothing from a plain prompt with greedy sampling")
	}
}

// TestRawChatGeneration repeats the raw drive, but with the prompt the chat path actually
// builds. It isolates the chat prompt from the engine: same model, same sampler, same loop.
func TestRawChatGeneration(t *testing.T) {
	path := os.Getenv("LH_MODEL")
	if path == "" {
		t.Skip("LH_MODEL not set")
	}
	Backend()
	defer BackendFree()
	if dir := os.Getenv("LH_BACKENDS"); dir != "" {
		LoadBackendsFrom(dir)
	} else {
		LoadBackends()
	}

	mp := DefaultModelParams()
	mp.NGPULayers = 0
	m, err := LoadModel(path, mp)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()

	tmpl, err := m.ChatTemplate()
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := ApplyChatTemplate(tmpl, []ChatMessage{
		{Role: "user", Content: "Write three sentences about the sea."},
	}, true)
	if err != nil {
		t.Fatal(err)
	}

	cp := DefaultContextParams()
	cp.NCtx = 2048
	cp.NBatch = 512
	cp.NSeqMax = 1
	cp.NThreads = 24
	cp.NThreadsBatch = 24
	ctx, err := NewContext(m, cp)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Free()

	vocab := m.Vocab()
	toks, err := vocab.Tokenize(prompt, true, true)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("chat prompt tokens (%d): %v", len(toks), toks)
	for _, tk := range toks {
		p, _ := vocab.TokenToPiece(tk, true)
		t.Logf("  %6d %q", tk, string(p))
	}

	b := NewBatch(512, 1)
	defer b.Free()
	for i, tk := range toks {
		if err := b.Add(tk, Pos(i), []SeqID{0}, i == len(toks)-1); err != nil {
			t.Fatal(err)
		}
	}
	if err := ctx.Decode(b); err != nil {
		t.Fatalf("prefill decode: %v", err)
	}

	smpl, err := NewSampler(SamplingParams{}, vocab.NTokens(), vocab)
	if err != nil {
		t.Fatal(err)
	}
	defer smpl.Free()

	var out strings.Builder
	pos := Pos(len(toks))
	for i := 0; i < 24; i++ {
		tk := smpl.Sample(ctx, -1)
		piece, _ := vocab.TokenToPiece(tk, true)
		t.Logf("step %d: token %d %q", i, tk, string(piece))
		if tk == vocab.EOS() {
			break
		}
		out.Write(piece)
		smpl.Accept(tk)
		b.Clear()
		if err := b.Add(tk, pos, []SeqID{0}, true); err != nil {
			t.Fatal(err)
		}
		if err := ctx.Decode(b); err != nil {
			t.Fatalf("decode step %d: %v", i, err)
		}
		pos++
	}
	t.Logf("CHATGENERATED>>>%s<<<", out.String())
}
