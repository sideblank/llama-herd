// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// build assembles a minimal GGUF in memory so parsing and the tied/untied decision are tested
// against known input, not only against whatever model happens to be on the box.
type builder struct{ b bytes.Buffer }

func (w *builder) u32(v uint32) { binary.Write(&w.b, binary.LittleEndian, v) }
func (w *builder) u64(v uint64) { binary.Write(&w.b, binary.LittleEndian, v) }
func (w *builder) str(s string) {
	w.u64(uint64(len(s)))
	w.b.WriteString(s)
}

func build(tensors []Tensor, kv map[string]string) []byte {
	w := &builder{}
	w.u32(magic)
	w.u32(3)
	w.u64(uint64(len(tensors)))
	w.u64(uint64(len(kv)))
	for k, v := range kv {
		w.str(k)
		w.u32(8) // string
		w.str(v)
	}
	for _, t := range tensors {
		w.str(t.Name)
		w.u32(uint32(len(t.Dims)))
		for _, d := range t.Dims {
			w.u64(d)
		}
		w.u32(t.Type)
		w.u64(t.Offset)
	}
	return w.b.Bytes()
}

func TestReadsTensorDirectory(t *testing.T) {
	raw := build([]Tensor{
		{Name: "token_embd.weight", Dims: []uint64{896, 151936}, Type: 8, Offset: 0},
		{Name: "blk.0.attn_q.weight", Dims: []uint64{896, 896}, Type: 12, Offset: 4096},
	}, map[string]string{"general.architecture": "qwen2"})

	f, err := read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if len(f.Tensors) != 2 {
		t.Fatalf("got %d tensors, want 2", len(f.Tensors))
	}
	e, ok := f.Find("token_embd.weight")
	if !ok {
		t.Fatal("token_embd.weight not found")
	}
	if got := e.Elements(); got != 896*151936 {
		t.Errorf("elements = %d, want %d", got, 896*151936)
	}
}

// The decision this package exists to make, tested both ways. Getting it backwards would bound a
// whole architecture on a false premise — or fail to bound one that should be.
func TestTiedDetectionBothWays(t *testing.T) {
	tied := build([]Tensor{
		{Name: "token_embd.weight", Dims: []uint64{896, 151936}, Type: 8},
	}, nil)
	untied := build([]Tensor{
		{Name: "token_embd.weight", Dims: []uint64{2048, 248320}, Type: 14},
		{Name: "output.weight", Dims: []uint64{2048, 248320}, Type: 14},
	}, nil)

	f, err := read(bytes.NewReader(tied))
	if err != nil {
		t.Fatal(err)
	}
	if got, why := f.TiedEmbeddings(); !got {
		t.Errorf("model without output.weight reported untied: %s", why)
	}

	f, err = read(bytes.NewReader(untied))
	if err != nil {
		t.Fatal(err)
	}
	if got, why := f.TiedEmbeddings(); got {
		t.Errorf("model with a separate output.weight reported tied: %s", why)
	}
}

// A model with no embedding tensor must not be reported either way — "cannot tell" is a distinct
// answer from "untied", and conflating them would silently license a latent-space experiment on a
// model nothing is known about.
func TestUnknownWhenNoEmbeddingTensor(t *testing.T) {
	raw := build([]Tensor{{Name: "blk.0.attn_q.weight", Dims: []uint64{4, 4}}}, nil)
	f, err := read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	tied, why := f.TiedEmbeddings()
	if tied {
		t.Error("reported tied with no token_embd.weight present")
	}
	if why == "" || !bytes.Contains([]byte(why), []byte("cannot tell")) {
		t.Errorf("reason should say it cannot tell, got %q", why)
	}
}

// Metadata carries the tokenizer vocabulary — hundreds of thousands of strings — so array skipping
// has to be right or the tensor table is read from the wrong offset and yields nonsense.
func TestSkipsArrayMetadata(t *testing.T) {
	w := &builder{}
	w.u32(magic)
	w.u32(3)
	w.u64(1) // tensors
	w.u64(1) // kv
	w.str("tokenizer.ggml.tokens")
	w.u32(9) // array
	w.u32(8) // of strings
	w.u64(3) // three of them
	w.str("a")
	w.str("bb")
	w.str("ccc")
	w.str("token_embd.weight")
	w.u32(2)
	w.u64(16)
	w.u64(32)
	w.u32(8)
	w.u64(0)

	f, err := read(bytes.NewReader(w.b.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	e, ok := f.Find("token_embd.weight")
	if !ok {
		t.Fatal("tensor table misread after an array — the skip is wrong")
	}
	if e.Dims[0] != 16 || e.Dims[1] != 32 {
		t.Errorf("dims = %v, want [16 32]", e.Dims)
	}
}

func TestRejectsNonGGUF(t *testing.T) {
	if _, err := read(bytes.NewReader([]byte("not a gguf file at all"))); err == nil {
		t.Error("expected a magic-number error")
	}
}

// A truncated file must error rather than return a partial directory that looks complete.
func TestTruncatedFileErrors(t *testing.T) {
	raw := build([]Tensor{
		{Name: "token_embd.weight", Dims: []uint64{896, 151936}, Type: 8},
		{Name: "output.weight", Dims: []uint64{896, 151936}, Type: 8},
	}, nil)
	if _, err := read(bytes.NewReader(raw[:len(raw)-12])); err == nil {
		t.Error("a truncated tensor table must error, not return what it managed to read")
	}
}
