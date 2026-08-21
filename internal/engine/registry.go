// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package engine

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
)

// Registry hosts several models on one host: one Engine each, each with its own resident
// weights and decode loop, sharing the GPU. This is the herd.
type Registry struct {
	mu      sync.RWMutex
	entries map[string]*entry
	wg      sync.WaitGroup
}

type entry struct {
	engine   *Engine
	renderer Renderer
	// err is set when the decode loop exits abnormally. A dead engine is refused
	// immediately: a request routed to one would otherwise queue for a loop that will
	// never tick again, turning an engine failure into a pile of hung requests.
	err error
}

var (
	// ErrUnknownModel means no engine is registered under that name.
	ErrUnknownModel = errors.New("registry: unknown model")
	// ErrModelDead means the model's decode loop exited and it can no longer serve.
	ErrModelDead = errors.New("registry: model is no longer running")
)

// NewRegistry creates an empty registry.
func NewRegistry() *Registry {
	return &Registry{entries: map[string]*entry{}}
}

// Add registers an engine under name. It is an error to reuse a name, because requests
// address models by name and a silent overwrite would route traffic to the wrong weights.
func (r *Registry) Add(name string, e *Engine, rend Renderer) error {
	if name == "" {
		return errors.New("registry: model name must not be empty")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.entries[name]; exists {
		return fmt.Errorf("registry: model %q already registered", name)
	}
	r.entries[name] = &entry{engine: e, renderer: rend}
	return nil
}

// Start runs every registered engine's decode loop until ctx is cancelled.
func (r *Registry) Start(ctx context.Context) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	for _, en := range r.entries {
		en := en
		r.wg.Add(1)
		go func() {
			defer r.wg.Done()
			err := en.engine.Run(ctx)
			if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
				r.mu.Lock()
				en.err = fmt.Errorf("%w: %v", ErrModelDead, err)
				r.mu.Unlock()
			}
		}()
	}
}

// Wait blocks until every decode loop has exited.
func (r *Registry) Wait() { r.wg.Wait() }

// Get returns the engine for name, refusing one whose loop has died.
func (r *Registry) Get(name string) (*Engine, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	en, ok := r.entries[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownModel, name)
	}
	if en.err != nil {
		return nil, en.err
	}
	return en.engine, nil
}

// Renderer returns the chat renderer for name, if it has one.
func (r *Registry) Renderer(name string) (Renderer, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	en, ok := r.entries[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownModel, name)
	}
	if en.renderer == nil {
		return nil, fmt.Errorf("model %q cannot render chat messages: it has no chat template", name)
	}
	return en.renderer, nil
}

// Names lists registered models in a stable order.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	out := make([]string, 0, len(r.entries))
	for name := range r.entries {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// Health reports each model's liveness, for a readiness probe that distinguishes "not
// listening" from "listening but the model behind it is gone".
func (r *Registry) Health() map[string]error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	out := make(map[string]error, len(r.entries))
	for name, en := range r.entries {
		out[name] = en.err
	}
	return out
}
