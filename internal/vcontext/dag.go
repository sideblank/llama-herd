// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"fmt"
	"sort"
	"strings"
)

// Task is one unit of work with its prerequisites.
//
// Extracted from the request before any generation, so ordering is enforced by a scheduler rather
// than asked of a model. A model asked to respect "do B after A" across a long context can drift or
// invent an intermediate step, and the failure is invisible — the output reads as a plan. A graph
// either sorts or it does not.
type Task struct {
	ID        string   `json:"id"`
	Desc      string   `json:"desc"`
	DependsOn []string `json:"depends_on"`
}

// Graph is an extracted set of tasks.
type Graph struct {
	Tasks []Task `json:"tasks"`
}

// Tier is a set of tasks whose dependencies are all satisfied by earlier tiers.
type Tier struct {
	Level int
	Tasks []Task
}

// CycleError names the tasks forming a circular dependency.
//
// Naming them is the point. "A cycle was detected" tells the caller nothing actionable in a graph
// of forty tasks; the cycle itself is the bug report.
type CycleError struct {
	Involved []string
}

func (e *CycleError) Error() string {
	return fmt.Sprintf("vcontext: circular dependency between %s — no ordering exists, so this "+
		"cannot be scheduled at all", strings.Join(e.Involved, " -> "))
}

// MissingDepError names a dependency that no task provides.
type MissingDepError struct {
	Task, Missing string
}

func (e *MissingDepError) Error() string {
	return fmt.Sprintf("vcontext: task %q depends on %q, which no task provides — the extraction "+
		"invented a prerequisite or dropped a task", e.Task, e.Missing)
}

// Sort resolves the graph into execution tiers by in-degree (Kahn's algorithm).
//
// Tiers are how the graph is reasoned about and reported. They are NOT how it is executed — see
// Schedule, which starts a task the moment its own dependencies are met rather than waiting for its
// whole tier. Executing tiers as barriers makes every tier wait for its slowest member, which
// converts an ordering constraint into a straggler problem.
func (g Graph) Sort() ([]Tier, error) {
	byID := map[string]Task{}
	var order []string
	for _, t := range g.Tasks {
		if t.ID == "" {
			return nil, fmt.Errorf("vcontext: a task has no id")
		}
		if _, dup := byID[t.ID]; dup {
			return nil, fmt.Errorf("vcontext: duplicate task id %q", t.ID)
		}
		byID[t.ID] = t
		order = append(order, t.ID)
	}

	indeg := map[string]int{}
	dependents := map[string][]string{}
	for _, t := range g.Tasks {
		for _, d := range t.DependsOn {
			if _, ok := byID[d]; !ok {
				return nil, &MissingDepError{Task: t.ID, Missing: d}
			}
			indeg[t.ID]++
			dependents[d] = append(dependents[d], t.ID)
		}
	}

	var tiers []Tier
	remaining := len(g.Tasks)
	seen := map[string]bool{}

	for level := 0; remaining > 0; level++ {
		var ready []string
		for _, id := range order {
			if !seen[id] && indeg[id] == 0 {
				ready = append(ready, id)
			}
		}
		if len(ready) == 0 {
			// Nothing can start and work remains: every unscheduled task is in or behind a
			// cycle. Report the ones still blocked.
			var stuck []string
			for _, id := range order {
				if !seen[id] {
					stuck = append(stuck, id)
				}
			}
			sort.Strings(stuck)
			return nil, &CycleError{Involved: stuck}
		}
		tier := Tier{Level: level}
		for _, id := range ready {
			seen[id] = true
			remaining--
			tier.Tasks = append(tier.Tasks, byID[id])
		}
		// Decrement only after the whole tier is taken, or a task could join the tier that
		// unblocked it.
		for _, id := range ready {
			for _, dep := range dependents[id] {
				indeg[dep]--
			}
		}
		tiers = append(tiers, tier)
	}
	return tiers, nil
}

// Independent reports tasks with no prerequisites and no dependents: background work that can run
// throughout without blocking or being blocked by the ordered chain.
func (g Graph) Independent() []Task {
	depended := map[string]bool{}
	for _, t := range g.Tasks {
		for _, d := range t.DependsOn {
			depended[d] = true
		}
	}
	var out []Task
	for _, t := range g.Tasks {
		if len(t.DependsOn) == 0 && !depended[t.ID] {
			out = append(out, t)
		}
	}
	return out
}

// CriticalPath is the longest chain of dependencies, which bounds how fast the graph can finish
// however many streams are available.
//
// Worth knowing before dispatch: a graph whose critical path is most of its tasks will not benefit
// from a wide herd, and that is a property of the request rather than of the deployment.
func (g Graph) CriticalPath() []string {
	byID := map[string]Task{}
	for _, t := range g.Tasks {
		byID[t.ID] = t
	}
	memo := map[string][]string{}
	var walk func(string) []string
	walk = func(id string) []string {
		if p, ok := memo[id]; ok {
			return p
		}
		memo[id] = nil // guards against a cycle; Sort reports it properly
		var best []string
		for _, d := range byID[id].DependsOn {
			if p := walk(d); len(p) > len(best) {
				best = p
			}
		}
		path := append(append([]string(nil), best...), id)
		memo[id] = path
		return path
	}
	var longest []string
	for _, t := range g.Tasks {
		if p := walk(t.ID); len(p) > len(longest) {
			longest = p
		}
	}
	return longest
}

// TaskGrammar constrains extraction to the shape the scheduler can consume.
//
// A grammar rather than a parse-and-hope: an extraction that returns prose, or a task without an
// id, is a scheduling failure discovered after GPU time has been spent. This makes the shape valid
// by construction.
const TaskGrammar = `root     ::= "{" ws "\"tasks\":" ws "[" ws task (ws "," ws task)* ws "]" ws "}"
task     ::= "{" ws "\"id\":" ws str "," ws "\"desc\":" ws str "," ws "\"depends_on\":" ws deps ws "}"
deps     ::= "[" ws "]" | "[" ws str (ws "," ws str)* ws "]"
str      ::= "\"" [^"\\]* "\""
ws       ::= [ \t\n]*`
