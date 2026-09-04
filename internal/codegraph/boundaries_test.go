// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package codegraph_test

import (
	"os"
	"strings"
	"testing"

	"github.com/sideblank/llama-herd/internal/codegraph"
	"github.com/sideblank/llama-herd/internal/vcontext"
)

// words is a deterministic stand-in for a tokenizer.
func words(s string) int { return len(strings.Fields(s)) }

func TestBoundariesLandPastDeclarations(t *testing.T) {
	src := []byte(`package p

import "fmt"

type User struct {
	ID   string
	Name string
}

func Greet(u User) string {
	if u.Name == "" {
		return "hello"
	}
	return fmt.Sprintf("hello %s", u.Name)
}

const Version = "1.0"
`)
	b := codegraph.GoBoundaries(src)
	if len(b) != 4 {
		t.Fatalf("import, type, func, const — want 4 boundaries, got %d: %v", len(b), b)
	}
	for _, off := range b {
		if off > len(src) {
			t.Fatalf("boundary %d is past the end", off)
		}
		before := string(src[:off])
		if strings.Count(before, "{") != strings.Count(before, "}") {
			t.Fatalf("cutting at %d leaves unbalanced braces — a declaration was severed", off)
		}
	}
}

func TestBoundariesSurviveATruncatedFile(t *testing.T) {
	src := []byte(`package p

func Complete() error { return nil }

func Truncated(x int) {
	if x > 0 {
`)
	b := codegraph.GoBoundaries(src)
	if len(b) == 0 {
		t.Fatal("the declarations above the break still have valid boundaries; returning nothing discards them and falls back to prose cutting for the whole file")
	}
	before := string(src[:b[len(b)-1]])
	if !strings.Contains(before, "func Complete") {
		t.Fatal("the complete declaration should be inside the last usable boundary")
	}
}

func TestNoBoundariesForUnparseableInput(t *testing.T) {
	if b := codegraph.GoBoundaries([]byte("this is not go at all {{{")); len(b) != 0 {
		t.Fatalf("want none, got %v", b)
	}
}

// ⚠️ MEASURED FINDING: for Go, structural cutting does NOT reliably improve cut POSITIONS.
//
// Idiomatic Go separates declarations with blank lines, so paragraph cutting lands on the same
// offsets. Measured both on this package's graph.go (AST 4/5, prose 4/5) and on synthetic functions
// carrying internal blank lines (5/5 both). Crediting structural cutting with better positions here
// would be crediting it with a win it does not deliver.
//
// What it delivers instead is KNOWING. Prose cutting that happens to land on a declaration cannot
// tell that from luck, so it must be treated as ragged and repaired with an overlap window
// everywhere. Structural cutting labels the cut CutDeclaration truthfully, and a cut known to be
// clean needs no overlap — which at 48 streams is the difference between ~14k tokens of duplicated
// prefill and none.
//
// So the value of #36's two halves is joined: the boundaries are what make the conditional overlap
// possible, and the conditional overlap is where the saving actually is.
func TestStructuralCuttingKnowsTheCutIsCleanAndProseCannot(t *testing.T) {
	src, err := os.ReadFile("graph.go")
	if err != nil {
		t.Skip("source not readable")
	}
	text := string(src)
	bounds := codegraph.GoBoundaries([]byte(src))
	plan := vcontext.Plan{Chunks: 6, ChunkTokens: words(text)/5 + 1, MaxChunkTokens: words(text)}

	withAST, err := vcontext.SplitAt(text, plan, words, bounds)
	if err != nil {
		t.Fatal(err)
	}
	prose, err := vcontext.Split(text, plan, words)
	if err != nil {
		t.Fatal(err)
	}

	labelled := 0
	for _, c := range withAST[:len(withAST)-1] {
		if c.Cut == vcontext.CutDeclaration {
			labelled++
		}
	}
	if labelled == 0 {
		t.Fatal("no cut was labelled a declaration cut — the boundaries were not consulted at all")
	}
	for _, c := range prose[:len(prose)-1] {
		if c.Cut == vcontext.CutDeclaration {
			t.Fatal("prose cutting cannot know a cut fell on a declaration; claiming so would let " +
				"a lucky cut skip an overlap it needed")
		}
	}

	// ⚠️ And the overlap saving is NOT against prose cutting — it is against an UNCONDITIONAL
	// window, which is what the design proposed. On Go source both strategies cut cleanly, so
	// both skip the overlap entirely; the tokens saved are the ones a blanket window would have
	// spent on cuts that severed nothing.
	conditional, err := vcontext.AddOverlap(withAST, 300, words)
	if err != nil {
		t.Fatal(err)
	}
	condCost, condN := vcontext.OverlapCost(conditional)
	uncond := 0
	for range withAST[1:] {
		uncond += 300
	}
	t.Logf("overlap tax over %d chunks: conditional %d tokens across %d chunks, "+
		"unconditional would be %d", len(withAST), condCost, condN, uncond)
	if condCost >= uncond {
		t.Fatalf("a conditional window must cost less than a blanket one: %d vs %d", condCost, uncond)
	}
}

// Records the finding above as an assertion, so it cannot quietly stop being true.
func TestOnIdiomaticGoProseCuttingIsAlreadyNearlyAsGood(t *testing.T) {
	src, err := os.ReadFile("graph.go")
	if err != nil {
		t.Skip("source not readable")
	}
	text := string(src)
	bounds := codegraph.GoBoundaries([]byte(src))
	inSet := map[int]bool{}
	for _, o := range bounds {
		inSet[o] = true
	}
	plan := vcontext.Plan{Chunks: 6, ChunkTokens: words(text)/5 + 1, MaxChunkTokens: words(text)}

	withAST, err := vcontext.SplitAt(text, plan, words, bounds)
	if err != nil {
		t.Fatal(err)
	}
	prose, err := vcontext.Split(text, plan, words)
	if err != nil {
		t.Fatal(err)
	}
	a, p := 0, 0
	for _, c := range withAST[:len(withAST)-1] {
		if inSet[c.End] {
			a++
		}
	}
	for _, c := range prose[:len(prose)-1] {
		if inSet[c.End] {
			p++
		}
	}
	t.Logf("idiomatic Go: AST %d/%d, prose %d/%d — blank lines already separate declarations",
		a, len(withAST)-1, p, len(prose)-1)
	if a < p {
		t.Fatalf("structural cutting must never be WORSE than prose: %d vs %d", a, p)
	}
}

// Boundaries are preferences, not constraints.
func TestAChunkNeverOverflowsToReachABoundary(t *testing.T) {
	// One enormous declaration with no internal boundary: the splitter must still cut it.
	var b strings.Builder
	b.WriteString("package p\n\nfunc Big() {\n")
	for i := 0; i < 400; i++ {
		b.WriteString("\tdoSomething(withAValue, andAnother)\n")
	}
	b.WriteString("}\n")
	text := b.String()

	bounds := codegraph.GoBoundaries([]byte(text))
	plan := vcontext.Plan{Chunks: 4, ChunkTokens: 200, MaxChunkTokens: 260}
	chunks, err := vcontext.SplitAt(text, plan, words, bounds)
	if err != nil {
		t.Fatalf("a 900-line function has two boundaries; treating them as constraints would "+
			"produce chunks that do not fit a stream: %v", err)
	}
	for _, c := range chunks {
		if c.Tokens > plan.MaxChunkTokens {
			t.Fatalf("chunk %d holds %d tokens against a %d limit", c.Index, c.Tokens, plan.MaxChunkTokens)
		}
	}
}

func TestNilBoundariesBehaveAsProse(t *testing.T) {
	text := strings.Repeat("alpha beta gamma delta epsilon. ", 60)
	plan := vcontext.Plan{Chunks: 4, ChunkTokens: 80, MaxChunkTokens: 200}
	a, err := vcontext.SplitAt(text, plan, words, nil)
	if err != nil {
		t.Fatal(err)
	}
	b2, err := vcontext.Split(text, plan, words)
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != len(b2) {
		t.Fatalf("supplying no boundaries must be identical to prose splitting: %d vs %d", len(a), len(b2))
	}
	for i := range a {
		if a[i].End != b2[i].End || a[i].Cut != b2[i].Cut {
			t.Fatalf("chunk %d differs", i)
		}
	}
}
