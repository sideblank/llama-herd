// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package dag holds graph algorithms shared by the task scheduler and the code symbol graph.
//
// Kept separate because the two callers want opposite things from a cycle. In a task graph a cycle
// means no valid ordering exists and the request cannot be run. In a code graph a cycle is ordinary
// — mutually recursive types, a Go package with two files referring to each other, a C++ header
// pair — and means those symbols must be generated *together* rather than that generation is
// impossible.
package dag

import "sort"

// Component is a strongly connected set of nodes: either one node, or a group that must be treated
// as a unit because each reaches the others.
type Component struct {
	Nodes []string
	// Level is its position in the topological order of the condensation.
	Level int
}

// Cyclic reports whether this component is a genuine cycle rather than a lone node.
func (c Component) Cyclic() bool { return len(c.Nodes) > 1 }

// SCC finds strongly connected components using Tarjan's algorithm.
//
// adj maps a node to the nodes it depends on. Nodes are returned in deterministic order so a plan
// built from them is reproducible — an unstable order here would make two runs of the same request
// assign work to different streams, and any measurement comparing them would be noise.
func SCC(nodes []string, adj map[string][]string) []Component {
	index := map[string]int{}
	low := map[string]int{}
	onStack := map[string]bool{}
	var stack []string
	next := 0
	var out [][]string

	// Iterative: a deep dependency chain in a large request would otherwise risk the stack, and
	// the recursion depth is attacker-influenced whenever the request is user text.
	type frame struct {
		node string
		i    int
	}

	for _, root := range nodes {
		if _, seen := index[root]; seen {
			continue
		}
		var frames []frame
		index[root] = next
		low[root] = next
		next++
		stack = append(stack, root)
		onStack[root] = true
		frames = append(frames, frame{node: root})

		for len(frames) > 0 {
			f := &frames[len(frames)-1]
			succs := adj[f.node]
			if f.i < len(succs) {
				w := succs[f.i]
				f.i++
				if _, seen := index[w]; !seen {
					index[w] = next
					low[w] = next
					next++
					stack = append(stack, w)
					onStack[w] = true
					frames = append(frames, frame{node: w})
				} else if onStack[w] {
					if index[w] < low[f.node] {
						low[f.node] = index[w]
					}
				}
				continue
			}
			// Done with this node.
			if low[f.node] == index[f.node] {
				var comp []string
				for {
					w := stack[len(stack)-1]
					stack = stack[:len(stack)-1]
					onStack[w] = false
					comp = append(comp, w)
					if w == f.node {
						break
					}
				}
				sort.Strings(comp)
				out = append(out, comp)
			}
			v := f.node
			frames = frames[:len(frames)-1]
			if len(frames) > 0 {
				p := frames[len(frames)-1].node
				if low[v] < low[p] {
					low[p] = low[v]
				}
			}
		}
	}

	comps := make([]Component, 0, len(out))
	for _, c := range out {
		comps = append(comps, Component{Nodes: c})
	}
	return comps
}

// Condense collapses each strongly connected component to a single node and topologically sorts the
// result, assigning each component a Level.
//
// The condensation of any directed graph is acyclic by construction, so this always succeeds — that
// is the point. A code graph with circular imports has no valid *per-file* ordering, but always has
// a valid *per-component* ordering, and the component is the unit that has to be generated in one
// pass.
func Condense(nodes []string, adj map[string][]string) []Component {
	comps := SCC(nodes, adj)
	owner := map[string]int{}
	for i, c := range comps {
		for _, n := range c.Nodes {
			owner[n] = i
		}
	}

	indeg := make([]int, len(comps))
	edges := make([]map[int]bool, len(comps))
	for i := range edges {
		edges[i] = map[int]bool{}
	}
	for n, deps := range adj {
		to, ok := owner[n]
		if !ok {
			continue
		}
		for _, d := range deps {
			from, ok := owner[d]
			if !ok || from == to {
				continue // intra-component edges are what the component absorbed
			}
			if !edges[from][to] {
				edges[from][to] = true
				indeg[to]++
			}
		}
	}

	// Kahn over components, ordered deterministically by first node.
	order := make([]int, len(comps))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return comps[order[a]].Nodes[0] < comps[order[b]].Nodes[0]
	})

	done := make([]bool, len(comps))
	var out []Component
	remaining := len(comps)
	for level := 0; remaining > 0; level++ {
		var ready []int
		for _, i := range order {
			if !done[i] && indeg[i] == 0 {
				ready = append(ready, i)
			}
		}
		if len(ready) == 0 {
			// Unreachable: a condensation cannot contain a cycle. Returning what is resolved
			// rather than looping forever, because a silent hang is the worse failure.
			break
		}
		for _, i := range ready {
			done[i] = true
			remaining--
			c := comps[i]
			c.Level = level
			out = append(out, c)
		}
		for _, i := range ready {
			for to := range edges[i] {
				indeg[to]--
			}
		}
	}
	return out
}
