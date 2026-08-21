// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package engine

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"sync"
)

// Request is one generation to run.
type Request struct {
	Prompt string
	// MaxTokens caps generated tokens. 0 means until EOS or context exhaustion.
	MaxTokens int
	// Stop ends generation when the accumulated output ends with any of these.
	Stop []string
	// Sampling overrides the model's configured sampling for this request only.
	// Nil keeps the model's defaults.
	Sampling *SamplingParams
	// Media carries encoded images or audio. Requires a backend implementing
	// MediaBackend, and the prompt must contain that backend's marker once per item.
	Media [][]byte
}

// Event is one item in a stream's output.
type Event struct {
	// Text is the newly decoded text, already buffered to a UTF-8 boundary.
	Text string
	// Done marks the final event. Reason says why.
	Done   bool
	Reason string
	Err    error
}

// Stop reasons.
const (
	ReasonEOS     = "stop"
	ReasonLength  = "length"
	ReasonStopSeq = "stop_sequence"
	ReasonContext = "context_exhausted"
	ReasonCancel  = "cancelled"
)

// Stream is the caller's handle on one running generation.
type Stream struct {
	Events <-chan Event
	cancel func()
}

// Close abandons the stream. The slot is released on the next tick.
func (s *Stream) Close() { s.cancel() }

// slot is one in-flight generation occupying a sequence in the context.
type slot struct {
	seq SeqID

	pending []Token // prompt tokens not yet fed
	pos     Pos     // next position in the sequence
	next    Token   // token to feed on the next decode tick
	primed  bool    // prefill finished; next holds a sampled token

	generated int
	maxTokens int
	stop      []string
	sampling  *SamplingParams
	// sampledOnce guards SetSampling so the chain is installed exactly once, before
	// this slot's first sample, rather than on every tick.
	sampledOnce bool

	out    chan Event
	ctx    context.Context
	cancel func()

	// batchIdx is where this slot's logits landed in the batch just decoded. It is only
	// meaningful when hasLogits is set; a media prefill produces logits at the last
	// position rather than at an index this loop assigned, which is why the two are
	// tracked separately instead of overloading -1 to mean both "none" and "last".
	batchIdx  int32
	hasLogits bool

	media     [][]byte
	prompt    string
	prefilled bool

	// draft holds the tokens proposed for this tick, and specIdx where each was staged in
	// the batch. Verification samples at those indices in order.
	draft   []Token
	specIdx []int32

	// buf accumulates bytes that are not yet a complete UTF-8 sequence, and text used
	// for stop-sequence matching.
	buf  []byte
	text string
}

// Engine drives one model. Exactly one goroutine — Run — touches the Backend.
type Engine struct {
	be Backend

	mu      sync.Mutex
	queue   []*slot
	free    []SeqID
	maxWait int

	wake chan struct{}

	drafter Drafter
	// observeFailed records that the drafter already reported an observation failure, so
	// the warning appears once rather than on every pass for the life of the process.
	observeFailed bool

	closed bool

	c counters
}

// Config tunes admission.
type Config struct {
	// MaxQueue bounds requests waiting for a slot. 0 means unbounded.
	MaxQueue int

	// Drafter enables speculative decoding when set. Each speculating stream consumes
	// several batch entries instead of one, so it trades batch capacity for the chance to
	// produce several tokens from a single pass.
	Drafter Drafter
}

// New creates an engine over be.
func New(be Backend, cfg Config) *Engine {
	n := int(be.NSeqMax())
	e := &Engine{
		be:      be,
		free:    make([]SeqID, 0, n),
		maxWait: cfg.MaxQueue,
		drafter: cfg.Drafter,
		wake:    make(chan struct{}, 1),
	}
	for i := 0; i < n; i++ {
		e.free = append(e.free, SeqID(i))
	}
	return e
}

// ErrQueueFull is returned when admission is refused rather than queued unboundedly.
var ErrQueueFull = errors.New("engine: queue full")

// Submit queues a request and returns a Stream. It does not block on capacity: the decode
// loop admits it when a sequence frees up.
func (e *Engine) Submit(ctx context.Context, req Request) (*Stream, error) {
	toks, err := e.be.Tokenize(req.Prompt, true)
	if err != nil {
		return nil, fmt.Errorf("tokenize: %w", err)
	}
	if len(toks) == 0 {
		return nil, errors.New("engine: empty prompt")
	}
	if budget := e.promptBudget(req.MaxTokens); uint32(len(toks)) > budget {
		return nil, fmt.Errorf("engine: prompt of %d tokens exceeds the per-sequence budget of %d "+
			"(context %d shared across %d slots, minus room to generate)",
			len(toks), budget, e.be.NCtxSeq(), e.be.NSeqMax())
	}

	if len(req.Media) > 0 {
		mb, ok := e.be.(MediaBackend)
		if !ok {
			return nil, errors.New("engine: this model cannot accept media")
		}
		if marker := mb.MediaMarker(); !strings.Contains(req.Prompt, marker) {
			return nil, fmt.Errorf("engine: prompt must contain the media marker %q once per "+
				"item, or the media is silently dropped and the model answers about nothing", marker)
		}
	}

	cctx, cancel := context.WithCancel(ctx)
	s := &slot{
		seq:       -1,
		pending:   toks,
		maxTokens: req.MaxTokens,
		stop:      req.Stop,
		sampling:  req.Sampling,
		media:     req.Media,
		prompt:    req.Prompt,
		out:       make(chan Event, 32),
		ctx:       cctx,
		cancel:    cancel,
		batchIdx:  -1,
	}

	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		cancel()
		return nil, errors.New("engine: closed")
	}
	if e.maxWait > 0 && len(e.queue) >= e.maxWait {
		e.mu.Unlock()
		cancel()
		return nil, ErrQueueFull
	}
	e.queue = append(e.queue, s)
	e.mu.Unlock()

	e.c.requests.Add(1)
	e.c.prompt.Add(uint64(len(toks)))

	e.signal()
	return &Stream{Events: s.out, cancel: cancel}, nil
}

// promptBudget is how many prompt tokens one sequence may hold, leaving room to generate.
//
// The check is against the PER-SEQUENCE context, not the total: the total is shared across
// every slot, so a prompt validated against it is admitted and then runs out of cells mid
// generation, which surfaces as an eviction rather than a clear rejection at submit time.
func (e *Engine) promptBudget(maxTokens int) uint32 {
	seq := e.be.NCtxSeq()
	if seq == 0 {
		seq = e.be.NCtx()
	}
	reserve := uint32(defaultOutputReserve)
	if maxTokens > 0 {
		reserve = uint32(maxTokens)
	}
	if reserve >= seq {
		return 0
	}
	return seq - reserve
}

// defaultOutputReserve is the room kept for generation when a request names no token limit.
const defaultOutputReserve = 512

func (e *Engine) signal() {
	select {
	case e.wake <- struct{}{}:
	default:
	}
}

// MediaMarker returns the placeholder this model's prompts must contain where media
// belongs, and whether the model accepts media at all.
func (e *Engine) MediaMarker() (string, bool) {
	mb, ok := e.be.(MediaBackend)
	if !ok {
		return "", false
	}
	return mb.MediaMarker(), true
}

// Run drives the decode loop until ctx is cancelled. It must be called exactly once, and it
// is the only goroutine permitted to touch the Backend.
func (e *Engine) Run(ctx context.Context) error {
	active := make(map[SeqID]*slot)

	for {
		select {
		case <-ctx.Done():
			e.drain(active, ReasonCancel)
			return ctx.Err()
		default:
		}

		e.admit(active)

		if len(active) == 0 {
			// Nothing to do. Wait for work rather than spinning.
			select {
			case <-ctx.Done():
				e.drain(active, ReasonCancel)
				return ctx.Err()
			case <-e.wake:
			}
			continue
		}

		if err := e.tick(active); err != nil {
			e.drain(active, ReasonCancel)
			return err
		}
	}
}

// admit moves queued requests into free sequences.
func (e *Engine) admit(active map[SeqID]*slot) {
	e.mu.Lock()
	defer e.mu.Unlock()

	for len(e.free) > 0 && len(e.queue) > 0 {
		s := e.queue[0]
		if s.ctx.Err() != nil { // cancelled while waiting
			e.queue = e.queue[1:]
			close(s.out)
			continue
		}
		e.queue = e.queue[1:]

		// Take the lowest free sequence rather than the most recently released one, so
		// slot assignment is predictable when reading logs or reproducing a failure.
		seq := e.free[0]
		e.free = e.free[1:]

		s.seq = seq
		s.sampledOnce = false
		active[seq] = s
		e.c.active.Add(1)

		// A context-predicting drafter wants the prompt, which is where most of its
		// hits come from.
		if sd, ok := e.drafter.(Seeder); ok && sd != nil {
			sd.Seed(seq, s.pending)
		}
	}
}

// tick builds one batch, decodes it, and routes the results.
//
// The batch is built in two passes: decode tokens first, then prefill with whatever budget
// remains. Filling with prefill first lets a long prompt consume the whole batch and starve
// — or worse, overrun — the already-running streams, which is a silent engine death rather
// than a clean error.
func (e *Engine) tick(active map[SeqID]*slot) error {
	e.be.BatchClear()
	budget := e.be.BatchCap()
	if budget <= 0 {
		return errors.New("engine: backend reports zero batch capacity")
	}

	for _, s := range active {
		s.batchIdx = -1
		s.hasLogits = false
	}

	// Media prefill happens outside the shared batch: it encodes and decodes internally,
	// then hands back the position to continue from.
	for seq, s := range active {
		if s.primed || s.prefilled || len(s.media) == 0 || s.ctx.Err() != nil {
			continue
		}
		mb, ok := e.be.(MediaBackend)
		if !ok {
			e.finish(active, seq, "", errors.New("engine: backend cannot accept media"))
			continue
		}
		newPast, err := mb.PrefillMedia(s.seq, 0, s.prompt, s.media, true)
		if err != nil {
			e.finish(active, seq, "", fmt.Errorf("media prefill: %w", err))
			continue
		}
		s.pos = newPast
		s.pending = nil
		s.prefilled = true
		s.primed = true
		// Logits sit on the final position of the prefill, not at an index this loop
		// assigned, so the sampler is told to read the last one.
		s.batchIdx = -1
		s.hasLogits = true
	}

	// Pass 1 — the active streams. Each stages its next token, and when a drafter is
	// present it also stages that draft's proposals so several positions are verified in
	// the same forward pass.
	//
	// Every staged entry consumes batch budget, drafts included. Counting only the real
	// token would let a speculating stream overrun the batch, which the backend rejects
	// and which kills the loop rather than degrading it.
	for _, s := range active {
		if !s.primed || s.ctx.Err() != nil {
			continue
		}
		if e.be.BatchLen() >= budget {
			break
		}

		s.draft = s.draft[:0]
		s.specIdx = s.specIdx[:0]

		idx := e.be.BatchLen()
		if err := e.be.BatchAdd(s.next, s.pos, s.seq, true); err != nil {
			break
		}
		s.batchIdx = idx
		s.hasLogits = true
		s.specIdx = append(s.specIdx, idx)
		s.pos++

		if e.drafter == nil {
			continue
		}
		room := int(budget - e.be.BatchLen())
		if n := min(e.drafter.MaxDraft(), room); n > 0 {
			proposed, err := e.drafter.Draft(s.seq, s.next, s.pos, n)
			if err != nil {
				// A failing drafter degrades to ordinary decoding rather than
				// failing the request: speculation is an optimisation.
				proposed = nil
			}
			for _, t := range proposed {
				i := e.be.BatchLen()
				if e.be.BatchAdd(t, s.pos, s.seq, true) != nil {
					break
				}
				s.draft = append(s.draft, t)
				s.specIdx = append(s.specIdx, i)
				s.pos++
				e.c.proposed.Add(1)
			}
		}
	}

	// Pass 2 — prefill, chunked into whatever budget is left.
	for _, s := range active {
		if s.primed || s.ctx.Err() != nil || len(s.pending) == 0 {
			continue
		}
		for len(s.pending) > 0 && e.be.BatchLen() < budget {
			last := len(s.pending) == 1
			idx := e.be.BatchLen()
			if err := e.be.BatchAdd(s.pending[0], s.pos, s.seq, last); err != nil {
				break
			}
			s.pending = s.pending[1:]
			s.pos++
			if last {
				s.batchIdx = idx
				s.hasLogits = true
				s.primed = true
			}
		}
	}

	if e.be.BatchLen() == 0 {
		// A media prefill may have produced logits without staging anything in the
		// shared batch, so there can still be slots to harvest.
		return e.harvest(active)
	}

	e.c.passes.Add(1)
	// A drafter predicting from the target's internal state must see every decode. This
	// runs after the pass below, not before, since there is nothing to observe until the
	// target has produced it.
	decodeErr := e.be.Decode()
	if ob, ok := e.drafter.(BatchObserver); ok && ob != nil && decodeErr == nil {
		// A failure here is not fatal — the target decoded fine and the answer is correct
		// — but it leaves the drafter predicting from stale state, which shows up only as
		// acceptance decaying. Swallowing it silently reproduces the failure the observer
		// exists to prevent, so say it once rather than every pass.
		if err := ob.ObserveDecode(); err != nil && !e.observeFailed {
			e.observeFailed = true
			log.Printf("engine: drafter could not observe a decode (%v) — speculation will "+
				"work from stale state and acceptance will fall; further such errors are silent", err)
		}
	}
	if err := decodeErr; err != nil {
		if errors.Is(err, ErrNoKVSlot) {
			// The cache is full, not broken. Evict the longest-running stream so the
			// rest make progress rather than deadlocking on a full cache.
			e.evictOne(active)
			return nil
		}
		return fmt.Errorf("decode: %w", err)
	}

	return e.harvest(active)
}

// harvest samples each slot that produced logits and emits its token.
func (e *Engine) harvest(active map[SeqID]*slot) error {
	for seq, s := range active {
		if !s.hasLogits {
			continue
		}
		if s.ctx.Err() != nil {
			e.finish(active, seq, ReasonCancel, nil)
			continue
		}

		if !s.sampledOnce {
			// Install per-request sampling on the decode-loop goroutine, which is the
			// only one permitted to touch the backend.
			if err := e.be.SetSampling(s.seq, s.sampling); err != nil {
				e.finish(active, seq, "", fmt.Errorf("sampling: %w", err))
				continue
			}
			s.sampledOnce = true
		}

		// Verify every position staged for this slot, in order. Without speculation
		// there is exactly one; with it, the first is the real token and the rest are
		// proposals that hold only while the target agrees.
		accepted, stop, reason, err := e.verify(s)
		if err != nil {
			e.finish(active, seq, "", err)
			continue
		}

		// Rewind the positions the target did not take. The drafts were written into
		// the cache to be checked, and leaving a rejected tail there would continue the
		// sequence from a state the model never chose.
		kept := accepted + 1
		if kept < len(s.specIdx) {
			s.pos -= Pos(len(s.specIdx) - kept)
			e.be.TrimSeq(s.seq, s.pos)
		}
		if e.drafter != nil && len(s.draft) > 0 {
			e.c.acceptedD.Add(uint64(accepted))
			_ = e.drafter.Accept(s.seq, accepted, s.next)
		}

		if stop != "" {
			e.finish(active, seq, stop, nil)
			continue
		}
		_ = reason

		switch {
		case s.hitStop():
			e.finish(active, seq, ReasonStopSeq, nil)
		case s.maxTokens > 0 && s.generated >= s.maxTokens:
			e.finish(active, seq, ReasonLength, nil)
		case uint32(s.pos) >= e.be.NCtxSeq():
			e.finish(active, seq, ReasonContext, nil)
		}
	}
	return nil
}

// verify samples each staged position and accepts the longest prefix the target agrees with.
//
// Returns how many drafted tokens were accepted, and a stop reason if generation ended part
// way through. The token at a divergence is the target's own choice, which is why a rejected
// draft still yields one real token: speculation never costs a step, it only sometimes saves
// several.
func (e *Engine) verify(s *slot) (accepted int, stop string, reason string, err error) {
	eos := e.be.EOS()

	for i, idx := range s.specIdx {
		tok, sErr := e.be.SampleAt(s.seq, idx)
		if sErr != nil {
			return accepted, "", "", fmt.Errorf("sample: %w", sErr)
		}

		if eos >= 0 && tok == eos {
			return accepted, ReasonEOS, "", nil
		}

		piece, pErr := e.be.Piece(tok)
		if pErr != nil {
			return accepted, "", "", fmt.Errorf("detokenize: %w", pErr)
		}

		s.next = tok
		s.generated++
		e.c.tokens.Add(1)
		if text, ok := s.consume(piece); ok {
			s.emit(Event{Text: text})
		}

		if s.hitStop() {
			return accepted, ReasonStopSeq, "", nil
		}
		if s.maxTokens > 0 && s.generated >= s.maxTokens {
			return accepted, ReasonLength, "", nil
		}

		// The next staged position holds a proposal, and it is only usable if the target
		// produced exactly that token here. On divergence the target's own token stands
		// and everything after it is discarded.
		if i < len(s.draft) {
			if s.draft[i] != tok {
				return accepted, "", "", nil
			}
			accepted++
		}
	}
	return accepted, "", "", nil
}

// consume appends piece and returns any text that is now a complete UTF-8 sequence.
// A multi-byte rune can span several tokens, so emitting each piece raw would produce
// mojibake at the boundary.
func (s *slot) consume(piece []byte) (string, bool) {
	s.buf = append(s.buf, piece...)
	cut := completeUTF8(s.buf)
	if cut == 0 {
		return "", false
	}
	text := string(s.buf[:cut])
	s.buf = append(s.buf[:0], s.buf[cut:]...)
	s.text += text
	return text, true
}

func (s *slot) hitStop() bool {
	for _, stop := range s.stop {
		if stop != "" && len(s.text) >= len(stop) && s.text[len(s.text)-len(stop):] == stop {
			return true
		}
	}
	return false
}

func (s *slot) emit(ev Event) {
	select {
	case s.out <- ev:
	case <-s.ctx.Done():
	}
}

// finish releases a slot's sequence and closes its stream.
func (e *Engine) finish(active map[SeqID]*slot, seq SeqID, reason string, err error) {
	s := active[seq]
	if s == nil {
		return
	}
	delete(active, seq)
	e.c.active.Add(-1)
	if e.drafter != nil {
		e.drafter.Release(s.seq)
	}
	if err != nil {
		e.c.failed.Add(1)
	}
	if reason == ReasonContext {
		e.c.evictions.Add(1)
	}
	e.be.FreeSeq(s.seq)

	s.emit(Event{Done: true, Reason: reason, Err: err})
	close(s.out)
	s.cancel()

	e.mu.Lock()
	e.free = append(e.free, s.seq)
	e.mu.Unlock()
	e.signal()
}

// evictOne frees the slot that has generated the most, on the grounds that it is closest to
// finishing anyway and its cells are the largest single reclaim available.
func (e *Engine) evictOne(active map[SeqID]*slot) {
	var victim *slot
	for _, s := range active {
		if victim == nil || s.generated > victim.generated {
			victim = s
		}
	}
	if victim != nil {
		e.finish(active, victim.seq, ReasonContext, nil)
	}
}

// drain ends every active and queued stream.
func (e *Engine) drain(active map[SeqID]*slot, reason string) {
	for seq := range active {
		e.finish(active, seq, reason, nil)
	}
	e.mu.Lock()
	q := e.queue
	e.queue = nil
	e.closed = true
	e.mu.Unlock()
	for _, s := range q {
		s.emit(Event{Done: true, Reason: reason})
		close(s.out)
		s.cancel()
	}
}

// completeUTF8 returns the length of the longest prefix of b that ends on a rune boundary.
func completeUTF8(b []byte) int {
	if len(b) == 0 {
		return 0
	}
	// Walk back at most 3 bytes looking for the start of a truncated sequence.
	for i := 1; i <= 4 && i <= len(b); i++ {
		c := b[len(b)-i]
		switch {
		case c&0x80 == 0: // ASCII: complete
			if i == 1 {
				return len(b)
			}
			return len(b)
		case c&0xC0 == 0x80: // continuation byte, keep walking back
			continue
		case c&0xE0 == 0xC0:
			if i >= 2 {
				return len(b)
			}
			return len(b) - i
		case c&0xF0 == 0xE0:
			if i >= 3 {
				return len(b)
			}
			return len(b) - i
		case c&0xF8 == 0xF0:
			if i >= 4 {
				return len(b)
			}
			return len(b) - i
		}
	}
	return len(b)
}
