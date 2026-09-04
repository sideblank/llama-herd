// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package codegraph

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"sort"
	"strings"
)

// GoDeclReader reads declarations out of generated Go using the standard library.
//
// Go first because it is what this project is written in, so the reader can be verified against
// real code without adding a dependency. It also settles the interface before tree-sitter arrives:
// if the contract holds for one language parsed natively, the multi-language path is a matter of
// swapping the parser, not redesigning the check.
type GoDeclReader struct{}

func (GoDeclReader) Lang() string { return "go" }

// Extract parses src and returns its top-level declarations.
//
// Partial results are returned when the source does not fully parse. Generated code frequently has
// a truncated final function or an unclosed brace — the stream hit its token limit — and the
// declarations above the break are still exactly what the dependent tier needs. Refusing the whole
// file over a fault in its last line would discard a working contract and turn a recoverable
// generation into a failed one.
func (GoDeclReader) Extract(unit string, src []byte) (SignatureSet, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "generated.go", src, parser.SkipObjectResolution|parser.AllErrors)
	if file == nil {
		return SignatureSet{}, fmt.Errorf("codegraph: %s did not parse at all: %w", unit, err)
	}

	set := SignatureSet{Unit: unit, Lang: "go", Sigs: map[string]Signature{}}
	render := func(n ast.Node) string {
		var b bytes.Buffer
		if perr := printer.Fprint(&b, fset, n); perr != nil {
			return ""
		}
		return strings.TrimSpace(b.String())
	}

	for _, d := range file.Decls {
		switch decl := d.(type) {
		case *ast.FuncDecl:
			if decl.Name == nil {
				continue
			}
			name := decl.Name.Name
			kind, recv := "func", ""
			if decl.Recv != nil && len(decl.Recv.List) > 0 {
				kind = "method"
				recv = strings.TrimLeft(render(decl.Recv.List[0].Type), "*")
				name = recv + "." + name
			}
			// Print the signature without the body: the contract is the shape, and a body
			// would spend a dependent stream's context on code it must not reimplement.
			sig := *decl
			sig.Body = nil
			sig.Doc = nil
			set.Sigs[name] = Signature{
				Symbol: name, Kind: kind, Recv: recv,
				Text: strings.TrimSuffix(render(&sig), "\n"),
			}

		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					if s.Name == nil {
						continue
					}
					kind := "type"
					if _, isIface := s.Type.(*ast.InterfaceType); isIface {
						kind = "interface"
					}
					set.Sigs[s.Name.Name] = Signature{
						Symbol: s.Name.Name, Kind: kind,
						Text: "type " + render(s),
					}
				case *ast.ValueSpec:
					k := "var"
					if decl.Tok == token.CONST {
						k = "const"
					}
					for _, n := range s.Names {
						if n.Name == "_" {
							continue
						}
						set.Sigs[n.Name] = Signature{
							Symbol: n.Name, Kind: k,
							Text: k + " " + render(s),
						}
					}
				}
			}
		}
	}
	return set, nil
}

// GoImports returns the paths a generated file imports, which is how a produced unit is checked
// against the edges the graph predicted. An import the graph did not know about means the
// extraction missed a dependency — the failure mode a cycle check cannot see.
func GoImports(src []byte) []string {
	fset := token.NewFileSet()
	file, _ := parser.ParseFile(fset, "generated.go", src, parser.ImportsOnly|parser.SkipObjectResolution)
	if file == nil {
		return nil
	}
	var out []string
	for _, im := range file.Imports {
		if im.Path != nil {
			out = append(out, strings.Trim(im.Path.Value, `"`))
		}
	}
	return out
}

// GoBoundaries returns byte offsets in src where a top-level declaration ends.
//
// These are the only cut points that can promise a declaration was not severed, which is what a
// judging pass needs: a chunk holding half a function is a chunk whose local pass reports on a
// fragment, confidently and with no way to know.
//
// Offsets are ascending and exclusive — each is the position just past a declaration's final byte,
// so cutting there keeps the whole declaration in the preceding chunk.
//
// Partial parses contribute what they got. Generated or truncated source frequently fails to parse
// at the very end, and the declarations above the break still have valid boundaries; returning
// nothing would discard them and fall back to prose cutting for the whole file.
func GoBoundaries(src []byte) []int {
	fset := token.NewFileSet()
	file, _ := parser.ParseFile(fset, "src.go", src, parser.SkipObjectResolution|parser.AllErrors)
	if file == nil {
		return nil
	}
	var out []int
	for _, d := range file.Decls {
		end := fset.Position(d.End()).Offset
		if end <= 0 || end > len(src) {
			continue
		}
		// Take the following newline with the declaration, so the next chunk does not open on a
		// blank line that belongs to the previous one.
		for end < len(src) && (src[end] == '\n' || src[end] == '\r') {
			end++
		}
		out = append(out, end)
	}
	sort.Ints(out)
	return out
}
