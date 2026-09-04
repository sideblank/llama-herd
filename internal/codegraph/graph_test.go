// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package codegraph

import (
	"strings"
	"testing"

	"github.com/sideblank/llama-herd/internal/dag"
)

func unit(id string, kind Kind, provides, requires []string) Unit {
	return Unit{ID: id, Kind: kind, Lang: "go", Provides: provides, Requires: requires}
}

func levelOf(p Plan, id string) int {
	for _, g := range p.Groups {
		for _, u := range g.Units {
			if u.ID == id {
				return g.Level
			}
		}
	}
	return -1
}

func TestImplComesAfterItsTypesAndInterface(t *testing.T) {
	g := Graph{Units: []Unit{
		unit("service", KindImpl, []string{"UserService"}, []string{"User", "UserRepository"}),
		unit("types", KindType, []string{"User"}, nil),
		unit("repo", KindInterface, []string{"UserRepository"}, nil),
		unit("itest", KindTest, nil, []string{"UserService"}),
	}}
	p, err := g.Build()
	if err != nil {
		t.Fatal(err)
	}
	if levelOf(p, "types") != 0 || levelOf(p, "repo") != 0 {
		t.Fatalf("types and interface are roots; got %d and %d", levelOf(p, "types"), levelOf(p, "repo"))
	}
	if levelOf(p, "service") <= levelOf(p, "types") {
		t.Fatal("an implementation generated before its types has imports that do not resolve")
	}
	if levelOf(p, "itest") <= levelOf(p, "service") {
		t.Fatal("the integration test must follow what it exercises")
	}
}

func TestStandaloneUtilityRunsInTheFirstLevel(t *testing.T) {
	g := Graph{Units: []Unit{
		unit("types", KindType, []string{"User"}, nil),
		unit("impl", KindImpl, []string{"Svc"}, []string{"User"}),
		unit("strhelper", KindUtility, []string{"StringHelper"}, nil),
	}}
	p, _ := g.Build()
	if levelOf(p, "strhelper") != 0 {
		t.Fatalf("a utility depending on nothing must not be held back — it would idle a stream that could be working (got level %d)", levelOf(p, "strhelper"))
	}
}

// The distinction between a code graph and a task graph.
func TestStdlibRequirementIsExternalNotMissing(t *testing.T) {
	g := Graph{Units: []Unit{
		unit("svc", KindImpl, []string{"Svc"}, []string{"context", "fmt", "User"}),
		unit("types", KindType, []string{"User"}, nil),
	}}
	p, err := g.Build()
	if err != nil {
		t.Fatalf("requiring the standard library must not be an error: %v", err)
	}
	ext := strings.Join(p.Resolution.External["svc"], ",")
	if !strings.Contains(ext, "context") || !strings.Contains(ext, "fmt") {
		t.Fatalf("context and fmt should be external requirements, got %q", ext)
	}
	if len(p.Resolution.Deps["svc"]) != 1 || p.Resolution.Deps["svc"][0] != "types" {
		t.Fatalf("only User resolves to a unit; got %v", p.Resolution.Deps["svc"])
	}
}

func TestAmbiguousSymbolIsWarnedNotGuessed(t *testing.T) {
	g := Graph{Units: []Unit{
		unit("a", KindType, []string{"User"}, nil),
		unit("b", KindType, []string{"User"}, nil),
		unit("svc", KindImpl, []string{"Svc"}, []string{"User"}),
	}}
	p, _ := g.Build()
	if len(p.Resolution.Ambiguous["User"]) != 2 {
		t.Fatalf("both providers must be recorded, got %v", p.Resolution.Ambiguous)
	}
	joined := strings.Join(p.Warnings, " ")
	if !strings.Contains(joined, "User") {
		t.Fatalf("an ambiguous symbol must warn — silently picking one yields code importing the wrong unit; warnings were %v", p.Warnings)
	}
}

// Circular imports are ordinary in code and must not be rejected the way a task cycle is.
func TestCircularImportsAreConsolidatedNotRejected(t *testing.T) {
	g := Graph{Units: []Unit{
		unit("node", KindType, []string{"Node"}, []string{"Edge"}),
		unit("edge", KindType, []string{"Edge"}, []string{"Node"}),
		unit("walk", KindImpl, []string{"Walk"}, []string{"Node"}),
	}}
	p, err := g.Build()
	if err != nil {
		t.Fatalf("a dependency cycle in code is not an error: %v", err)
	}
	var cyc *Group
	for i := range p.Groups {
		if p.Groups[i].Cyclic {
			cyc = &p.Groups[i]
		}
	}
	if cyc == nil {
		t.Fatal("the mutually recursive types must form one consolidated group")
	}
	if len(cyc.Units) != 2 {
		t.Fatalf("both units belong to the group, got %d", len(cyc.Units))
	}
	if levelOf(p, "walk") <= cyc.Level {
		t.Fatal("the consumer must still follow the consolidated group")
	}
	if !strings.Contains(strings.Join(p.Warnings, " "), "circular") {
		t.Fatal("consolidation changes how the work is dispatched and must be reported")
	}
}

func TestSelfProvidedRequirementIsNotASelfEdge(t *testing.T) {
	g := Graph{Units: []Unit{unit("a", KindType, []string{"T"}, []string{"T"})}}
	p, err := g.Build()
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Resolution.Deps["a"]) != 0 {
		t.Fatalf("a unit cannot depend on itself, got %v", p.Resolution.Deps["a"])
	}
	if len(p.Groups) != 1 || p.Groups[0].Cyclic {
		t.Fatal("a lone unit referencing its own symbol is not a cycle")
	}
}

func TestSymbolNormalisationMatchesTheFormsModelsEmit(t *testing.T) {
	for _, form := range []string{"User", "*User", "[]User", "models.User", "pkg/models.User"} {
		g := Graph{Units: []Unit{
			unit("types", KindType, []string{"User"}, nil),
			unit("svc", KindImpl, []string{"Svc"}, []string{form}),
		}}
		p, _ := g.Build()
		if len(p.Resolution.Deps["svc"]) != 1 {
			t.Fatalf("%q should resolve to the User provider, got %v", form, p.Resolution.Deps["svc"])
		}
	}
}

func TestKindPriorNeverPullsAUnitEarlierThanItsEdges(t *testing.T) {
	// A "utility" that genuinely depends on a type must not be dragged to level 0 by its kind.
	g := Graph{Units: []Unit{
		unit("types", KindType, []string{"User"}, nil),
		unit("helper", KindUtility, []string{"Fmt"}, []string{"User"}),
	}}
	p, _ := g.Build()
	if levelOf(p, "helper") <= levelOf(p, "types") {
		t.Fatal("a real edge must outrank the kind heuristic")
	}
}

func TestTestWithNoStatedDepsStillRunsLast(t *testing.T) {
	g := Graph{Units: []Unit{
		unit("types", KindType, []string{"User"}, nil),
		unit("tests", KindTest, nil, nil),
	}}
	p, _ := g.Build()
	if levelOf(p, "tests") == 0 {
		t.Fatal("a test that declared no requirements would otherwise generate against nothing")
	}
}

func TestWidthReportsWhatTheRequestCanUse(t *testing.T) {
	g := Graph{Units: []Unit{
		unit("a", KindType, []string{"A"}, nil),
		unit("b", KindType, []string{"B"}, nil),
		unit("c", KindType, []string{"C"}, nil),
		unit("d", KindImpl, []string{"D"}, []string{"A", "B", "C"}),
	}}
	p, _ := g.Build()
	if p.Width() != 3 {
		t.Fatalf("three roots can generate at once, got width %d", p.Width())
	}
}

func TestDuplicateUnitIDRejected(t *testing.T) {
	g := Graph{Units: []Unit{unit("a", KindType, nil, nil), unit("a", KindImpl, nil, nil)}}
	if _, err := g.Build(); err == nil {
		t.Fatal("duplicate ids make every edge referring to them ambiguous")
	}
}

func TestEmptyGraph(t *testing.T) {
	p, err := Graph{}.Build()
	if err != nil || len(p.Groups) != 0 || p.Width() != 0 {
		t.Fatalf("empty graph should plan cleanly, got %+v %v", p, err)
	}
}

// --- SCC ---

func TestCondenseHandlesADeepChainWithoutRecursion(t *testing.T) {
	const n = 20000
	var nodes []string
	adj := map[string][]string{}
	for i := 0; i < n; i++ {
		id := string(rune('a')) + string(rune(i))
		_ = id
	}
	nodes = nil
	for i := 0; i < n; i++ {
		nodes = append(nodes, itoa(i))
		if i > 0 {
			adj[itoa(i)] = []string{itoa(i - 1)}
		}
	}
	comps := dag.Condense(nodes, adj)
	if len(comps) != n {
		t.Fatalf("a chain has no cycles: want %d components, got %d", n, len(comps))
	}
	// A recursive Tarjan would have overflowed the stack well before here; a deep chain is
	// reachable from user text, so the depth must not be a limit.
	if comps[0].Level != 0 {
		t.Fatal("the chain head is level 0")
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

func TestCondenseIsDeterministic(t *testing.T) {
	nodes := []string{"a", "b", "c", "d"}
	adj := map[string][]string{"b": {"a"}, "c": {"a"}, "d": {"b", "c"}}
	first := dag.Condense(nodes, adj)
	for i := 0; i < 20; i++ {
		got := dag.Condense(nodes, adj)
		if len(got) != len(first) {
			t.Fatal("component count changed between runs")
		}
		for j := range got {
			if got[j].Nodes[0] != first[j].Nodes[0] || got[j].Level != first[j].Level {
				t.Fatal("an unstable order would assign work to different streams on identical input, making any A/B measurement noise")
			}
		}
	}
}

func TestSCCFindsAMultiNodeCycle(t *testing.T) {
	nodes := []string{"a", "b", "c", "d"}
	adj := map[string][]string{"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"a"}}
	comps := dag.SCC(nodes, adj)
	var found bool
	for _, c := range comps {
		if c.Cyclic() {
			found = true
			if len(c.Nodes) != 3 {
				t.Fatalf("the cycle is a,b,c; got %v", c.Nodes)
			}
		}
	}
	if !found {
		t.Fatal("no cycle detected")
	}
}
