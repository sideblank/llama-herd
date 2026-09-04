// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package gguf reads the tensor directory of a GGUF file.
//
// Deliberately pure Go and deliberately partial: it reads the header and tensor table, not the
// weights. That is enough to answer structural questions about a model — which tensors exist, how
// large they are — without loading it, without a GPU, and without adding cgo surface.
//
// The question that motivated it: does a model tie its input and output embeddings? That decides
// whether a hidden state can carry more than one token of information, which decides whether a
// whole architecture is worth building. It is answerable from the tensor names alone.
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const magic = 0x46554747 // "GGUF" little-endian

// Tensor is one entry in the file's tensor directory.
type Tensor struct {
	Name   string
	Dims   []uint64
	Type   uint32
	Offset uint64
}

// Elements is how many values the tensor holds.
func (t Tensor) Elements() uint64 {
	n := uint64(1)
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

// File is a GGUF file's structure, without its weights.
type File struct {
	Version uint32
	Tensors []Tensor
}

// Open reads the header and tensor directory.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return read(f)
}

func read(r io.Reader) (*File, error) {
	br := &reader{r: r}

	if m := br.u32(); m != magic {
		return nil, fmt.Errorf("gguf: not a GGUF file (magic %#x)", m)
	}
	version := br.u32()
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", version)
	}
	nTensors := br.u64()
	nKV := br.u64()
	if br.err != nil {
		return nil, br.err
	}
	// A corrupt header can claim an absurd count; refuse rather than allocate on it.
	if nTensors > 1<<20 || nKV > 1<<20 {
		return nil, fmt.Errorf("gguf: implausible header (%d tensors, %d metadata keys)",
			nTensors, nKV)
	}

	// Metadata is skipped rather than parsed — the values are reachable through the library's
	// own API, and this package exists for the tensor table.
	for i := uint64(0); i < nKV && br.err == nil; i++ {
		br.str()
		br.value(br.u32())
	}
	if br.err != nil {
		return nil, fmt.Errorf("gguf: reading metadata: %w", br.err)
	}

	out := &File{Version: version, Tensors: make([]Tensor, 0, nTensors)}
	for i := uint64(0); i < nTensors && br.err == nil; i++ {
		t := Tensor{Name: br.str()}
		nd := br.u32()
		if nd > 8 {
			return nil, fmt.Errorf("gguf: tensor %q claims %d dimensions", t.Name, nd)
		}
		for d := uint32(0); d < nd; d++ {
			t.Dims = append(t.Dims, br.u64())
		}
		t.Type = br.u32()
		t.Offset = br.u64()
		out.Tensors = append(out.Tensors, t)
	}
	if br.err != nil {
		return nil, fmt.Errorf("gguf: reading tensor table: %w", br.err)
	}
	return out, nil
}

// Find returns the named tensor.
func (f *File) Find(name string) (Tensor, bool) {
	for _, t := range f.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return Tensor{}, false
}

// TiedEmbeddings reports whether the model reuses its input embedding matrix as the output
// projection, and the tensor names it decided from.
//
// Why it matters: with tied embeddings the logits are E·h, so the natural map from a final hidden
// state back to input-embedding space is Eᵀ·softmax(logits) — the expected embedding under the
// predicted next token. A hidden state then carries roughly ONE TOKEN of information, which bounds
// what any architecture built on passing hidden states around can achieve.
//
// Untied, no such reduction exists and the state may carry considerably more.
func (f *File) TiedEmbeddings() (tied bool, why string) {
	_, hasEmbd := f.Find("token_embd.weight")
	out, hasOut := f.Find("output.weight")
	switch {
	case !hasEmbd:
		return false, "no token_embd.weight — cannot tell"
	case hasOut:
		return false, fmt.Sprintf("output.weight present (%v), separate from token_embd.weight",
			out.Dims)
	default:
		return true, "no output.weight — the input embedding is reused as the output projection"
	}
}

// reader reads little-endian values, latching the first error so callers need not check each read.
type reader struct {
	r   io.Reader
	err error
	buf [8]byte
}

func (b *reader) fill(n int) []byte {
	if b.err != nil {
		return b.buf[:n]
	}
	if _, err := io.ReadFull(b.r, b.buf[:n]); err != nil {
		b.err = err
	}
	return b.buf[:n]
}

func (b *reader) u32() uint32 { return binary.LittleEndian.Uint32(b.fill(4)) }
func (b *reader) u64() uint64 { return binary.LittleEndian.Uint64(b.fill(8)) }

func (b *reader) str() string {
	n := b.u64()
	if b.err != nil {
		return ""
	}
	if n > 1<<20 {
		b.err = fmt.Errorf("gguf: implausible string length %d", n)
		return ""
	}
	s := make([]byte, n)
	if _, err := io.ReadFull(b.r, s); err != nil {
		b.err = err
		return ""
	}
	return string(s)
}

// value skips one metadata value of the given type.
func (b *reader) value(t uint32) {
	if b.err != nil {
		return
	}
	switch t {
	case 0, 1, 7: // uint8, int8, bool
		b.fill(1)
	case 2, 3: // uint16, int16
		b.fill(2)
	case 4, 5, 6: // uint32, int32, float32
		b.fill(4)
	case 10, 11, 12: // uint64, int64, float64
		b.fill(8)
	case 8: // string
		b.str()
	case 9: // array
		et := b.u32()
		n := b.u64()
		if b.err != nil {
			return
		}
		if n > 1<<28 {
			b.err = fmt.Errorf("gguf: implausible array length %d", n)
			return
		}
		for i := uint64(0); i < n && b.err == nil; i++ {
			b.value(et)
		}
	default:
		b.err = fmt.Errorf("gguf: unknown metadata value type %d", t)
	}
}
