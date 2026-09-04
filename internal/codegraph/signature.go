// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package codegraph

import (
	"fmt"
	"sort"
	"strings"
)

// Signature is one exported declaration, as exact source text.
//
// Text, not a latent vector, and this is the one place the choice is not close. Signature drift —
// tier 0 declaring GetUser(id string) while tier 1 implements GetUser(id int) — is the failure the
// whole tiering exists to prevent, and it turns on the precise bytes of a declaration. Forwarding a
// hidden state and letting the implementation stream reconstruct the signature from it reintroduces
// exactly the drift being designed out, in a form no reader can see.
type Signature struct {
	Symbol string
	Kind   string // "type", "func", "method", "interface", "const", "var"
	Recv   string // receiver type for methods, empty otherwise
	Text   string // the declaration as written, without a body
}

// SignatureSet is the contract a tier publishes to the tiers below it.
type SignatureSet struct {
	Unit string
	Lang string
	Sigs map[string]Signature
}

// Contract renders the set as the exact text to place in a dependent stream's prompt.
//
// Rendered from the parsed declarations rather than by pasting the whole generated file: an
// implementation stream given the full source of its dependency spends context on function bodies
// it must not reimplement, and at 8,874 tokens per stream that budget is the scarce thing.
func (s SignatureSet) Contract() string {
	if len(s.Sigs) == 0 {
		return ""
	}
	keys := make([]string, 0, len(s.Sigs))
	for k := range s.Sigs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	fmt.Fprintf(&b, "// Declared in %s. Use these exactly; do not redeclare or alter them.\n", s.Unit)
	for _, k := range keys {
		b.WriteString(s.Sigs[k].Text)
		b.WriteString("\n")
	}
	return b.String()
}

// DeclReader parses generated source and reports what it actually declared.
//
// An interface because the parser is per-language and the checking logic is not. The Go
// implementation uses the standard library; other languages arrive via tree-sitter, whose place in
// the pipeline is here — parsing code that now exists — and not at request time, when the request
// is prose and there is no syntax tree to build.
type DeclReader interface {
	Lang() string
	Extract(unit string, src []byte) (SignatureSet, error)
}

// DriftKind classifies a mismatch between what a tier promised and what a later tier produced.
type DriftKind string

const (
	// DriftMissing: the contract declared a symbol the implementation never defined.
	DriftMissing DriftKind = "missing"
	// DriftChanged: the symbol exists with a different signature. The dangerous one — the code
	// often still parses, and only a compiler or a caller finds it.
	DriftChanged DriftKind = "changed"
	// DriftRedeclared: the implementation redeclared a type its contract already provided,
	// which compiles in some languages and produces two incompatible types in others.
	DriftRedeclared DriftKind = "redeclared"
)

// Drift is one mismatch.
type Drift struct {
	Kind     DriftKind
	Symbol   string
	Promised string
	Observed string
}

func (d Drift) Error() string {
	switch d.Kind {
	case DriftMissing:
		return fmt.Sprintf("%s: declared as %q but never implemented", d.Symbol, d.Promised)
	case DriftRedeclared:
		return fmt.Sprintf("%s: redeclared as %q when the contract already provides %q",
			d.Symbol, d.Observed, d.Promised)
	default:
		return fmt.Sprintf("%s: contract says %q, implementation says %q",
			d.Symbol, d.Promised, d.Observed)
	}
}

// CheckDrift compares a contract against what an implementation actually produced.
//
// Injecting the exact declarations into the dependent stream reduces drift; it does not prevent it,
// because nothing constrains a model to honour text in its prompt. The check is what makes the
// guarantee real, and it is cheap: parsing a generated file costs microseconds against the seconds
// of GPU time that produced it.
//
// required names the symbols the implementation was asked to implement. Symbols in the contract but
// not required are dependencies it merely uses, and their absence is expected rather than drift.
func CheckDrift(contract SignatureSet, impl SignatureSet, required []string) []Drift {
	req := map[string]bool{}
	for _, r := range required {
		req[norm(r)] = true
	}
	var out []Drift
	for name, promised := range contract.Sigs {
		observed, present := impl.Sigs[name]
		switch {
		case !present:
			if req[norm(name)] {
				out = append(out, Drift{Kind: DriftMissing, Symbol: name, Promised: promised.Text})
			}
		case !sameSig(promised, observed):
			kind := DriftChanged
			if promised.Kind == "type" && observed.Kind == "type" && !req[norm(name)] {
				kind = DriftRedeclared
			}
			out = append(out, Drift{
				Kind: kind, Symbol: name,
				Promised: promised.Text, Observed: observed.Text,
			})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Symbol < out[j].Symbol })
	return out
}

// sameSig compares declarations ignoring only formatting, never content.
//
// Whitespace is normalised because a model reformats freely and that is not drift. Parameter names
// are NOT normalised away: a renamed parameter is usually harmless, but distinguishing harmless
// from meaningful needs the call sites, which are in another stream. Reporting it and letting a
// caller decide is better than deciding wrongly and silently here.
func sameSig(a, b Signature) bool {
	return a.Kind == b.Kind && a.Recv == b.Recv && squash(a.Text) == squash(b.Text)
}

func squash(s string) string { return strings.Join(strings.Fields(s), " ") }
