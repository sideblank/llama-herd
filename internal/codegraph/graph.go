// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package codegraph turns a code generation request into a symbol dependency graph and a
// generation plan.
//
// The ordering problem in code is stricter than in prose. A summary generated out of order reads
// oddly; an implementation generated before its interface has imports that do not resolve,
// signatures that do not match, and types that were never declared. None of that is visible in the
// generated text — it is visible only to a compiler nobody has run yet.
package codegraph

import (
	"fmt"
	"sort"
	"strings"

	"github.com/sideblank/llama-herd/internal/dag"
)

// Kind is what a unit produces. It supplies an ordering prior when the request states no explicit
// dependency, and nothing more: a real edge always wins over a kind.
type Kind string

const (
	KindType      Kind = "type"      // structs, data models, config, schemas
	KindInterface Kind = "interface" // traits, interfaces, headers, protocols
	KindImpl      Kind = "impl"      // business logic, handlers, query builders
	KindTest      Kind = "test"      // unit and integration tests
	KindUtility   Kind = "utility"   // standalone helpers, docs, config files
)

// rank is the fallback ordering when no symbol edge connects two units. Utilities rank with types
// because they depend on nothing by definition — holding them back would idle streams that could
// be working.
var rank = map[Kind]int{KindType: 0, KindInterface: 0, KindUtility: 0, KindImpl: 1, KindTest: 2}

// Unit is one file or symbol group to generate.
type Unit struct {
	ID       string   `json:"id"`
	Path     string   `json:"path"`
	Lang     string   `json:"lang"`
	Kind     Kind     `json:"kind"`
	Desc     string   `json:"desc"`
	Provides []string `json:"provides"`
	Requires []string `json:"requires"`
}

// Graph is the extracted set of units.
type Graph struct {
	Units []Unit `json:"units"`
}

// Resolution is the outcome of turning Requires into edges.
type Resolution struct {
	// Deps maps a unit to the units it depends on.
	Deps map[string][]string
	// External lists requirements that no unit provides.
	//
	// NOT an error, and this is the difference between a code graph and a task graph. A task
	// depending on a task that does not exist is a broken extraction. A unit requiring `context`
	// or `std::vector` is requiring the standard library, which is the common case — treating it
	// as a missing node would reject almost every real request.
	External map[string][]string
	// Ambiguous lists symbols provided by more than one unit.
	//
	// A real defect rather than a curiosity: the graph cannot know which unit an edge should
	// point at, and choosing silently produces code that imports the wrong one and still looks
	// correct.
	Ambiguous map[string][]string
	// SelfSatisfied lists units that require symbols they also provide. Harmless, and recorded
	// only so it is not counted as an edge — a unit cannot depend on itself.
	SelfSatisfied map[string][]string
}

// Resolve matches each unit's Requires against every unit's Provides.
func (g Graph) Resolve() (Resolution, error) {
	providers := map[string][]string{}
	seen := map[string]bool{}
	for _, u := range g.Units {
		if u.ID == "" {
			return Resolution{}, fmt.Errorf("codegraph: a unit has no id")
		}
		if seen[u.ID] {
			return Resolution{}, fmt.Errorf("codegraph: duplicate unit id %q", u.ID)
		}
		seen[u.ID] = true
		for _, p := range u.Provides {
			providers[norm(p)] = append(providers[norm(p)], u.ID)
		}
	}

	r := Resolution{
		Deps:          map[string][]string{},
		External:      map[string][]string{},
		Ambiguous:     map[string][]string{},
		SelfSatisfied: map[string][]string{},
	}
	for _, u := range g.Units {
		depSet := map[string]bool{}
		for _, req := range u.Requires {
			owners := providers[norm(req)]
			switch {
			case len(owners) == 0:
				r.External[u.ID] = append(r.External[u.ID], req)
			case len(owners) == 1 && owners[0] == u.ID:
				r.SelfSatisfied[u.ID] = append(r.SelfSatisfied[u.ID], req)
			default:
				if len(owners) > 1 {
					sorted := append([]string(nil), owners...)
					sort.Strings(sorted)
					r.Ambiguous[req] = sorted
				}
				for _, o := range owners {
					if o != u.ID {
						depSet[o] = true
					}
				}
			}
		}
		if len(depSet) > 0 {
			var deps []string
			for d := range depSet {
				deps = append(deps, d)
			}
			sort.Strings(deps)
			r.Deps[u.ID] = deps
		}
	}
	return r, nil
}

// norm makes symbol matching tolerant of the forms a model actually emits: `pkg.User`, `User`,
// `*User`, `[]User` all name the same symbol.
//
// Deliberately narrow. Matching too eagerly invents edges that serialise work for no reason, and a
// wrong edge is invisible — it just makes generation slower and no test fails.
func norm(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimLeft(s, "*&")
	s = strings.TrimPrefix(s, "[]")
	if i := strings.LastIndexAny(s, ".:/"); i >= 0 && i < len(s)-1 {
		s = s[i+1:]
	}
	return s
}

// Group is one generation unit in the plan: usually a single Unit, occasionally several that must
// be produced together.
type Group struct {
	Level int
	Units []Unit
	// Cyclic marks a group formed by a dependency cycle rather than by being a lone unit.
	//
	// Circular imports are not an error in code the way they are in a task list. Two mutually
	// recursive types have no valid per-file ordering and a perfectly good joint one, so the
	// cycle is consolidated into a single generation pass rather than rejected.
	Cyclic bool
}

// Plan is the full generation schedule.
type Plan struct {
	Groups     []Group
	Resolution Resolution
	// Warnings names things that will not stop generation but will probably produce wrong code.
	Warnings []string
}

// Levels returns the groups bucketed by level.
func (p Plan) Levels() [][]Group {
	if len(p.Groups) == 0 {
		return nil
	}
	max := 0
	for _, g := range p.Groups {
		if g.Level > max {
			max = g.Level
		}
	}
	out := make([][]Group, max+1)
	for _, g := range p.Groups {
		out[g.Level] = append(out[g.Level], g)
	}
	return out
}

// Width is the largest number of groups that can be generated at once, which is what the stream
// budget is compared against. A width of 3 on a 48-stream deployment means the request, not the
// hardware, is the limit.
func (p Plan) Width() int {
	w := 0
	for _, lvl := range p.Levels() {
		if len(lvl) > w {
			w = len(lvl)
		}
	}
	return w
}

// Build resolves the graph and produces a generation plan.
func (g Graph) Build() (Plan, error) {
	res, err := g.Resolve()
	if err != nil {
		return Plan{}, err
	}

	byID := map[string]Unit{}
	var ids []string
	for _, u := range g.Units {
		byID[u.ID] = u
		ids = append(ids, u.ID)
	}
	sort.Strings(ids)

	adj := map[string][]string{}
	for id, deps := range res.Deps {
		adj[id] = deps
	}

	comps := dag.Condense(ids, adj)

	plan := Plan{Resolution: res}
	for _, c := range comps {
		gr := Group{Level: c.Level, Cyclic: c.Cyclic()}
		for _, id := range c.Nodes {
			gr.Units = append(gr.Units, byID[id])
		}
		plan.Groups = append(plan.Groups, gr)
	}

	// Apply the kind prior only where the symbol graph left a unit unordered. A test with no
	// declared requirements would otherwise land in level 0 beside the types it is meant to
	// exercise, and generate against nothing.
	plan.applyKindPrior()

	for sym, owners := range res.Ambiguous {
		plan.Warnings = append(plan.Warnings, fmt.Sprintf(
			"symbol %q is provided by %s — the graph cannot tell which one a dependent should "+
				"import, and picking silently yields code that references the wrong unit",
			sym, strings.Join(owners, " and ")))
	}
	for _, gr := range plan.Groups {
		if gr.Cyclic {
			var names []string
			for _, u := range gr.Units {
				names = append(names, u.ID)
			}
			plan.Warnings = append(plan.Warnings, fmt.Sprintf(
				"circular dependency between %s — consolidated into one generation pass, which "+
					"is correct for code but means these units share a stream and their combined "+
					"output must fit one context",
				strings.Join(names, ", ")))
		}
	}
	sort.Strings(plan.Warnings)
	return plan, nil
}

// applyKindPrior pushes units whose level came out lower than their kind implies. It never pulls a
// unit earlier: a real dependency edge always outranks a heuristic about what tests usually need.
func (p *Plan) applyKindPrior() {
	for i := range p.Groups {
		want := 0
		for _, u := range p.Groups[i].Units {
			if r, ok := rank[u.Kind]; ok && r > want {
				want = r
			}
		}
		if want > p.Groups[i].Level {
			p.Groups[i].Level = want
		}
	}
	sort.SliceStable(p.Groups, func(a, b int) bool { return p.Groups[a].Level < p.Groups[b].Level })
}
