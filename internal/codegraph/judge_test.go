// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package codegraph

import (
	"strings"
	"testing"
)

func def(chunk int, sym string, arity int, sig string) Assertion {
	return Assertion{Chunk: chunk, Symbol: sym, Role: Defines, Arity: arity, Sig: sig}
}
func use(chunk int, sym string, arity int, sig string) Assertion {
	return Assertion{Chunk: chunk, Symbol: sym, Role: Uses, Arity: arity, Sig: sig}
}

// The worked example: chunk 3 calls DB.Query with 3 args, chunk 18 defines it taking 2.
func TestArityMismatchAcrossChunks(t *testing.T) {
	v := CrossCheck(48, []Assertion{
		use(3, "DB.Query", 3, ""),
		def(18, "DB.Query", 2, ""),
	}, nil)
	if v.NoConflictsFound() {
		t.Fatal("the mismatch neither chunk could see must be found")
	}
	c := v.Conflicts[0]
	if c.Kind != ArityMismatch {
		t.Fatalf("want arity mismatch, got %s", c.Kind)
	}
	s := c.String()
	if !strings.Contains(s, "chunk 3") || !strings.Contains(s, "chunk 18") {
		t.Fatalf("both chunks must be named so the caller knows where to look: %q", s)
	}
}

func TestSignatureMismatchAcrossChunks(t *testing.T) {
	v := CrossCheck(4, []Assertion{
		def(1, "GetUser", 1, "GetUser(id string) (*User, error)"),
		use(2, "GetUser", 1, "GetUser(id int) (*User, error)"),
	}, nil)
	if v.NoConflictsFound() || v.Conflicts[0].Kind != SignatureMismatch {
		t.Fatalf("same arity, different types — the drift that still compiles-looking: %+v", v.Conflicts)
	}
}

func TestFormattingIsNotAConflict(t *testing.T) {
	v := CrossCheck(2, []Assertion{
		def(0, "F", 1, "F(a  string)  error"),
		use(1, "F", 1, "F(a string) error"),
	}, nil)
	if !v.NoConflictsFound() {
		t.Fatalf("whitespace differences are reformatting, not drift: %v", v.Conflicts)
	}
}

func TestUseWithNoDefinitionIsReported(t *testing.T) {
	v := CrossCheck(3, []Assertion{use(1, "Missing", 2, "")}, nil)
	if v.NoConflictsFound() || v.Conflicts[0].Kind != Undefined {
		t.Fatalf("want undefined, got %+v", v.Conflicts)
	}
}

func TestTwoChunksDefiningTheSameSymbolDifferently(t *testing.T) {
	v := CrossCheck(4, []Assertion{
		def(1, "Config", 0, "type Config struct{ A int }"),
		def(2, "Config", 0, "type Config struct{ B string }"),
	}, nil)
	if v.NoConflictsFound() || v.Conflicts[0].Kind != Redefined {
		t.Fatalf("neither local pass can see the other; the aggregator is the only place this shows: %+v", v.Conflicts)
	}
}

func TestIdenticalRedefinitionIsNotAConflict(t *testing.T) {
	v := CrossCheck(4, []Assertion{
		def(1, "Config", 0, "type Config struct{ A int }"),
		def(2, "Config", 0, "type Config struct{ A int }"),
	}, nil)
	if !v.NoConflictsFound() {
		t.Fatalf("an overlap window legitimately reports the same declaration twice: %v", v.Conflicts)
	}
}

// The correction that matters most.
func TestSilentChunksPoisonTheVerdict(t *testing.T) {
	v := CrossCheck(48, []Assertion{def(0, "A", 0, "type A struct{}")}, nil)
	if !v.NoConflictsFound() {
		t.Fatal("no contradiction was reported")
	}
	if v.Coverage.Complete() {
		t.Fatal("47 chunks returned nothing; that cannot be a complete result")
	}
	s := v.Summary()
	if !strings.Contains(s, "NOT judged") || !strings.Contains(s, "partial") {
		t.Fatalf("a verdict over unexamined regions must say so — otherwise 'no conflicts' and 'nothing was checked' print identically: %q", s)
	}
}

func TestSummaryAlwaysCarriesCoverage(t *testing.T) {
	v := CrossCheck(2, []Assertion{def(0, "A", 0, "x"), use(1, "A", 0, "x")}, nil)
	if !v.NoConflictsFound() {
		t.Fatal("expected clean")
	}
	s := v.Summary()
	for _, want := range []string{"no cross-chunk conflicts", "2/2 chunks", "symbols"} {
		if !strings.Contains(s, want) {
			t.Fatalf("coverage must be inseparable from the result; %q lacks %q", s, want)
		}
	}
}

func TestUnassertedSymbolsAreReported(t *testing.T) {
	v := CrossCheck(1, []Assertion{def(0, "A", 0, "x")}, []string{"A", "B", "C"})
	if len(v.Coverage.Unasserted) != 2 {
		t.Fatalf("B and C were declared and never mentioned, so nothing was checked against them: %v", v.Coverage.Unasserted)
	}
	if v.Coverage.Complete() {
		t.Fatal("unmentioned declarations make the result partial")
	}
}

func TestNoVerdictTypeExposesABooleanCalledCorrect(t *testing.T) {
	// Guards the naming, which is the whole enforcement: a caller reaching for v.Correct()
	// should not find it, because the cross-check cannot answer that question.
	var v Verdict
	_ = v.NoConflictsFound()
	s := v.Summary()
	if strings.Contains(strings.ToLower(s), "correct") {
		t.Fatalf("the summary must not use the word 'correct' — it converts absence of evidence into a claim: %q", s)
	}
}

func TestSymbolFormsAreNormalisedInAssertions(t *testing.T) {
	v := CrossCheck(2, []Assertion{
		def(0, "models.User", 0, "type User struct{}"),
		use(1, "*User", 0, "type User struct{}"),
	}, nil)
	if !v.NoConflictsFound() {
		t.Fatalf("the same symbol written two ways must not read as undefined: %v", v.Conflicts)
	}
}

func TestZeroArityIsNotComparedAsAMismatch(t *testing.T) {
	// Arity 0 means "the chunk did not report one", not "takes no arguments". Treating an
	// unreported field as a claim manufactures conflicts on every partial assertion.
	v := CrossCheck(2, []Assertion{def(0, "F", 2, ""), use(1, "F", 0, "")}, nil)
	if !v.NoConflictsFound() {
		t.Fatalf("an absent arity is not a contradiction: %v", v.Conflicts)
	}
}

func TestEmptyRun(t *testing.T) {
	v := CrossCheck(0, nil, nil)
	if !v.NoConflictsFound() || !v.Coverage.Complete() {
		t.Fatal("nothing dispatched, nothing silent")
	}
}

func TestAssertionGrammarConstrainsTheAggregatorInputs(t *testing.T) {
	for _, want := range []string{"symbol", "role", "arity", "sig", "defines", "uses"} {
		if !strings.Contains(AssertionGrammar, want) {
			t.Fatalf("the grammar must require %q — a field the aggregator reads but the grammar leaves optional fails only after 48 streams have run", want)
		}
	}
}

// --- Go declaration reader + drift ---

func TestGoDeclReaderReadsDeclarations(t *testing.T) {
	src := []byte(`package p
type User struct { ID string }
type Repo interface { Get(id string) (*User, error) }
func New(r Repo) *Svc { return nil }
type Svc struct{}
func (s *Svc) Get(id string) (*User, error) { return nil, nil }
const Version = "1"
`)
	set, err := GoDeclReader{}.Extract("types.go", src)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"User", "Repo", "New", "Svc", "Svc.Get", "Version"} {
		if _, ok := set.Sigs[want]; !ok {
			t.Fatalf("missing %q; got %v", want, keys(set))
		}
	}
	if set.Sigs["Repo"].Kind != "interface" {
		t.Fatalf("Repo is an interface, got %q", set.Sigs["Repo"].Kind)
	}
	if strings.Contains(set.Sigs["New"].Text, "return nil") {
		t.Fatal("the contract must carry the signature without the body — a dependent stream given function bodies spends its context on code it must not reimplement")
	}
}

func TestGoDeclReaderReturnsWhatItGotFromTruncatedOutput(t *testing.T) {
	// A stream that hit its token limit mid-function. The declarations above the break are
	// still a usable contract.
	src := []byte(`package p
type User struct { ID string }
func Good() error { return nil }
func Truncated(x int) {
	if x > 0 {
`)
	set, err := GoDeclReader{}.Extract("partial.go", src)
	if err != nil {
		t.Fatalf("partial source must not be a hard failure: %v", err)
	}
	if _, ok := set.Sigs["User"]; !ok {
		t.Fatalf("declarations before the break must survive; got %v", keys(set))
	}
	if _, ok := set.Sigs["Good"]; !ok {
		t.Fatal("a complete function before the truncation must be extracted")
	}
}

func TestCheckDriftCatchesTheStringIntCase(t *testing.T) {
	contract, _ := GoDeclReader{}.Extract("iface.go", []byte(`package p
type User struct{}
func GetUser(id string) (*User, error) { return nil, nil }
`))
	impl, _ := GoDeclReader{}.Extract("impl.go", []byte(`package p
func GetUser(id int) (*User, error) { return nil, nil }
`))
	drift := CheckDrift(contract, impl, []string{"GetUser"})
	if len(drift) == 0 {
		t.Fatal("GetUser(id string) implemented as GetUser(id int) is the exact failure tiering exists to prevent")
	}
	if drift[0].Kind != DriftChanged {
		t.Fatalf("want changed, got %s", drift[0].Kind)
	}
	if !strings.Contains(drift[0].Error(), "string") || !strings.Contains(drift[0].Error(), "int") {
		t.Fatalf("the message must show both sides: %q", drift[0].Error())
	}
}

func TestCheckDriftFlagsAnUnimplementedRequirement(t *testing.T) {
	contract, _ := GoDeclReader{}.Extract("i.go", []byte("package p\nfunc A() {}\nfunc B() {}\n"))
	impl, _ := GoDeclReader{}.Extract("x.go", []byte("package p\nfunc A() {}\n"))
	drift := CheckDrift(contract, impl, []string{"A", "B"})
	if len(drift) != 1 || drift[0].Kind != DriftMissing || drift[0].Symbol != "B" {
		t.Fatalf("B was required and never implemented: %+v", drift)
	}
}

func TestCheckDriftIgnoresContractSymbolsMerelyUsed(t *testing.T) {
	contract, _ := GoDeclReader{}.Extract("i.go", []byte("package p\ntype User struct{}\nfunc Impl() {}\n"))
	impl, _ := GoDeclReader{}.Extract("x.go", []byte("package p\nfunc Impl() {}\n"))
	drift := CheckDrift(contract, impl, []string{"Impl"})
	if len(drift) != 0 {
		t.Fatalf("User is a dependency the unit uses, not one it was asked to implement — its absence is expected: %+v", drift)
	}
}

func TestCheckDriftAcceptsReformatting(t *testing.T) {
	contract, _ := GoDeclReader{}.Extract("i.go", []byte("package p\nfunc A(x int,y int) error { return nil }\n"))
	impl, _ := GoDeclReader{}.Extract("x.go", []byte("package p\nfunc A(x int, y int) error {\n\treturn nil\n}\n"))
	if d := CheckDrift(contract, impl, []string{"A"}); len(d) != 0 {
		t.Fatalf("reformatting is not drift: %+v", d)
	}
}

func TestContractRendersDeterministically(t *testing.T) {
	set, _ := GoDeclReader{}.Extract("t.go", []byte("package p\ntype B struct{}\ntype A struct{}\nfunc C() {}\n"))
	first := set.Contract()
	for i := 0; i < 10; i++ {
		if set.Contract() != first {
			t.Fatal("an unstable contract changes the prompt between runs and makes any comparison noise")
		}
	}
	if !strings.Contains(first, "do not redeclare") {
		t.Fatal("the contract must tell the dependent stream what to do with it")
	}
}

func TestGoImportsFound(t *testing.T) {
	got := GoImports([]byte("package p\nimport (\n\t\"context\"\n\t\"example.com/x\"\n)\n"))
	if len(got) != 2 || got[0] != "context" {
		t.Fatalf("got %v", got)
	}
}

func keys(s SignatureSet) []string {
	var out []string
	for k := range s.Sigs {
		out = append(out, k)
	}
	return out
}
