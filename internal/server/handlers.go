package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"bedrock-simple/internal/anthropic"
	"bedrock-simple/internal/bedrock"
	"bedrock-simple/internal/convert"
	"bedrock-simple/internal/logx"
	"bedrock-simple/internal/openai"
	"bedrock-simple/internal/store"
)

const maxRequestBytes = 64 << 20

func readBody(r *http.Request, into any) error {
	raw, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBytes))
	if err != nil {
		return fmt.Errorf("read request body: %w", err)
	}
	if err := json.Unmarshal(raw, into); err != nil {
		return fmt.Errorf("invalid JSON body: %w", err)
	}
	return nil
}

func keyID(k *store.APIKey) string {
	if k == nil {
		return ""
	}
	return k.ID
}

// applyEffortToConverse translates the hub's effort level into the extended
// thinking block Anthropic models on Converse expect. Mantle models read
// ReasoningEffort directly, so they need nothing here.
func applyEffortToConverse(body *bedrock.ConverseRequest, modelID string) {
	effort := body.ReasoningEffort
	if effort == "" || !convert.IsAnthropicModel(modelID) {
		return
	}
	budget := bedrock.BudgetForEffort(effort)
	if budget == 0 {
		delete(body.AdditionalModelRequestFields, "thinking")
		return
	}
	if body.AdditionalModelRequestFields == nil {
		body.AdditionalModelRequestFields = map[string]any{}
	}
	body.AdditionalModelRequestFields["thinking"] = map[string]any{
		"type":          "enabled",
		"budget_tokens": budget,
	}
	if ic := body.InferenceConfig; ic != nil {
		ic.Temperature = nil
		ic.TopP = nil
		if ic.MaxTokens == nil || *ic.MaxTokens <= budget {
			ic.MaxTokens = bedrock.Ptr(budget + 4096)
		}
	}
}

// sseWriter emits server-sent events and flushes each one immediately. It is
// mutex-guarded because the keepalive goroutine also writes to it.
type sseWriter struct {
	mu      sync.Mutex
	w       http.ResponseWriter
	flusher http.Flusher
	begun   bool
	wrote   bool
	last    time.Time
}

func newSSEWriter(w http.ResponseWriter) (*sseWriter, bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, false
	}
	// last starts now so the first heartbeat is timed from the request, not
	// from the first upstream event, which can be half a minute away.
	return &sseWriter{w: w, flusher: flusher, last: time.Now()}, true
}

// written reports whether any bytes have reached the client, which makes a
// failure non-retryable.
func (s *sseWriter) written() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.wrote
}

func (s *sseWriter) begin() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.beginLocked()
}

func (s *sseWriter) beginLocked() {
	// The heartbeat may open the stream concurrently with the first event.
	if s.begun {
		return
	}
	s.begun = true
	s.w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	s.w.Header().Set("Cache-Control", "no-cache")
	s.w.Header().Set("Connection", "keep-alive")
	s.w.Header().Set("X-Accel-Buffering", "no")
	s.w.WriteHeader(http.StatusOK)
	// Committing 200 rules out a JSON error response from here on.
	s.wrote = true
	s.last = time.Now()
	s.flusher.Flush()
}

// send writes one event. An empty name omits the event: line (OpenAI style).
func (s *sseWriter) send(name string, data any) error {
	raw, err := json.Marshal(data)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if name != "" {
		if _, err := fmt.Fprintf(s.w, "event: %s\n", name); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", raw); err != nil {
		return err
	}
	s.markWrittenLocked()
	return nil
}

func (s *sseWriter) raw(line string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	fmt.Fprint(s.w, line)
	s.markWrittenLocked()
}

func (s *sseWriter) markWrittenLocked() {
	s.wrote = true
	s.last = time.Now()
	s.flusher.Flush()
}

// keepAlive sends an idle heartbeat until the returned stop is called.
// Reasoning models can go minutes before the first token, so the heartbeat also
// runs before anything has been written: a client that sees no bytes at all
// cannot tell a thinking model from a dead connection. The first beat waits
// longer, because it commits a 200 and costs the ability to report an upstream
// rejection as a clean HTTP error. stop waits for the goroutine to exit so no
// heartbeat can land after the final event.
func (s *sseWriter) keepAlive(ctx context.Context, warmup, every time.Duration, beat func()) func() {
	done := make(chan struct{})
	finished := make(chan struct{})
	var once sync.Once

	go func() {
		defer close(finished)
		// Polling far more often than the interval bounds the real silence at
		// `every`. Ticking at `every` instead allows almost twice that, because
		// a write landing just after a tick is only noticed at the following one.
		t := time.NewTicker(heartbeatResolution(every))
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-done:
				return
			case <-t.C:
				s.mu.Lock()
				threshold := every
				if !s.wrote {
					threshold = warmup
				}
				idle := time.Since(s.last) >= threshold
				s.mu.Unlock()
				if idle {
					beat()
				}
			}
		}
	}()

	return func() {
		once.Do(func() { close(done) })
		<-finished
	}
}

// keepAliveInterval is well inside the idle tolerance of clients and proxies.
// Claude Code was observed abandoning a stream after 18.8s of silence, so this
// leaves room for a beat to land first.
const keepAliveInterval = 10 * time.Second

// firstKeepAlive outlasts a wrong-route rejection on a large prompt, which was
// measured at 18.3s for a 2MB body, so those still surface as a clean HTTP
// error. Routes are remembered across restarts, so paying it is rare.
const firstKeepAlive = 20 * time.Second

// heartbeatResolution polls often enough that the observed silence never much
// exceeds the interval itself.
func heartbeatResolution(every time.Duration) time.Duration {
	step := every / 10
	if step < time.Millisecond {
		return time.Millisecond
	}
	if step > time.Second {
		return time.Second
	}
	return step
}

// logStreamFailure reports a mid-stream failure. A cancelled context means the
// client hung up, which is routine and not worth an error line.
func logStreamFailure(r *http.Request, modelID string, err error) {
	if clientHungUp(r, err) {
		logx.Debugf("%s model=%s client disconnected mid-stream", r.URL.Path, modelID)
		return
	}
	logx.Errorf("%s model=%s stream aborted after partial output: %v", r.URL.Path, modelID, err)
}

// clientHungUp reports a disconnect by the caller rather than an upstream fault.
func clientHungUp(r *http.Request, err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(r.Context().Err(), context.Canceled)
}

// ------------------------------------------------------ OpenAI chat completions

func (h *Handler) handleChatCompletions(w http.ResponseWriter, r *http.Request, key *store.APIKey) {
	var req openai.ChatRequest
	if err := readBody(r, &req); err != nil {
		logx.Warnf("POST %s bad request: %v", r.URL.Path, err)
		writeAPIError(w, errStyleOpenAI, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	modelID, effort := h.registry.ResolveWithEffort(req.Model)
	body, err := convert.OpenAIToConverse(&req, modelID, store.DefaultMaxTokens())
	if err != nil {
		logx.Warnf("POST %s bad request model=%s: %v", r.URL.Path, req.Model, err)
		writeAPIError(w, errStyleOpenAI, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	// An explicit -thinking-<level> suffix beats whatever the body asked for.
	if effort != "" {
		body.ReasoningEffort = effort
	}
	applyEffortToConverse(body, modelID)

	tr := newTrace(r, req.Model, modelID, req.Stream)
	tr.request(body, effort)

	if req.Stream {
		h.streamChat(w, r, key, &req, modelID, body, tr)
		return
	}

	stopWatch := tr.watch(r.Context())
	var resp *bedrock.ConverseResponse
	resp, err = h.callUpstream(r, modelID, body, tr)
	stopWatch()
	tr.finish(err)
	if err != nil {
		store.RecordUsage(keyID(key), 0, 0, true)
		upstreamError(w, errStyleOpenAI, r, modelID, err)
		return
	}

	store.RecordUsage(keyID(key), int64(resp.Usage.InputTokens), int64(resp.Usage.OutputTokens), false)
	writeJSON(w, http.StatusOK, convert.ConverseToOpenAI(resp, newID("chatcmpl-"), req.Model, time.Now().Unix()))
}

func (h *Handler) streamChat(w http.ResponseWriter, r *http.Request, key *store.APIKey, req *openai.ChatRequest, modelID string, body *bedrock.ConverseRequest, tr *trace) {
	sse, ok := newSSEWriter(w)
	if !ok {
		writeAPIError(w, errStyleOpenAI, http.StatusInternalServerError, "api_error", "streaming unsupported by this server")
		return
	}

	conv := convert.NewOpenAIStream(newID("chatcmpl-"), req.Model, time.Now().Unix())
	// OpenAI has no ping event, so an SSE comment is the portable heartbeat.
	stopBeat := sse.keepAlive(r.Context(), firstKeepAlive, keepAliveInterval, func() {
		sse.begin()
		sse.raw(": keepalive\n\n")
	})
	defer stopBeat()
	stopWatch := tr.watch(r.Context())
	defer stopWatch()

	// Heartbeats carry no answer, so a route may still be retried after one.
	var sentContent atomic.Bool

	err := h.streamUpstream(r, modelID, body,
		sentContent.Load,
		func(ev bedrock.StreamEvent) error {
			tr.event(ev)
			for _, chunk := range conv.Handle(ev) {
				sse.begin()
				sentContent.Store(true)
				if err := sse.send("", chunk); err != nil {
					return err
				}
			}
			return nil
		}, tr)

	stopWatch()
	tr.finish(err)

	if err != nil {
		if !sse.written() {
			store.RecordUsage(keyID(key), 0, 0, true)
			upstreamError(w, errStyleOpenAI, r, modelID, err)
			return
		}
		logStreamFailure(r, modelID, err)
		_ = sse.send("", openai.NewError("api_error", "upstream_error", err.Error()))
		sse.raw("data: [DONE]\n\n")
		store.RecordUsage(keyID(key), 0, 0, !clientHungUp(r, err))
		return
	}

	stopBeat()
	if !sse.written() {
		sse.begin()
	}
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		if chunk := conv.UsageChunk(); chunk != nil {
			_ = sse.send("", chunk)
		}
	}
	sse.raw("data: [DONE]\n\n")

	if u := conv.Usage; u != nil {
		store.RecordUsage(keyID(key), int64(u.InputTokens), int64(u.OutputTokens), false)
	} else {
		store.RecordUsage(keyID(key), 0, 0, false)
	}
}

// handleCountTokens answers the token-count probe Claude Code makes before
// each turn to budget its context.
func (h *Handler) handleCountTokens(w http.ResponseWriter, r *http.Request, _ *store.APIKey) {
	var req anthropic.Request
	if err := readBody(r, &req); err != nil {
		logx.Warnf("POST %s bad request: %v", r.URL.Path, err)
		writeAPIError(w, errStyleAnthropic, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	modelID, _ := h.registry.ResolveWithEffort(req.Model)
	body, err := convert.AnthropicToConverse(&req, modelID, store.DefaultMaxTokens())
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]int{"input_tokens": 0})
		return
	}
	writeJSON(w, http.StatusOK, map[string]int{"input_tokens": convert.EstimateRequestTokens(body)})
}

// --------------------------------------------------------- Anthropic messages

func (h *Handler) handleMessages(w http.ResponseWriter, r *http.Request, key *store.APIKey) {
	var req anthropic.Request
	if err := readBody(r, &req); err != nil {
		logx.Warnf("POST %s bad request: %v", r.URL.Path, err)
		writeAPIError(w, errStyleAnthropic, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	modelID, effort := h.registry.ResolveWithEffort(req.Model)
	body, err := convert.AnthropicToConverse(&req, modelID, store.DefaultMaxTokens())
	if err != nil {
		logx.Warnf("POST %s bad request model=%s: %v", r.URL.Path, req.Model, err)
		writeAPIError(w, errStyleAnthropic, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if effort != "" {
		body.ReasoningEffort = effort
	}
	applyEffortToConverse(body, modelID)

	tr := newTrace(r, req.Model, modelID, req.Stream)
	tr.request(body, effort)

	if req.Stream {
		h.streamMessages(w, r, key, &req, modelID, body, tr)
		return
	}

	stopWatch := tr.watch(r.Context())
	var resp *bedrock.ConverseResponse
	resp, err = h.callUpstream(r, modelID, body, tr)
	stopWatch()
	tr.finish(err)
	if err != nil {
		store.RecordUsage(keyID(key), 0, 0, true)
		upstreamError(w, errStyleAnthropic, r, modelID, err)
		return
	}

	store.RecordUsage(keyID(key), int64(resp.Usage.InputTokens), int64(resp.Usage.OutputTokens), false)
	writeJSON(w, http.StatusOK, convert.ConverseToAnthropic(resp, newID("msg_"), req.Model))
}

func (h *Handler) streamMessages(w http.ResponseWriter, r *http.Request, key *store.APIKey, req *anthropic.Request, modelID string, body *bedrock.ConverseRequest, tr *trace) {
	sse, ok := newSSEWriter(w)
	if !ok {
		writeAPIError(w, errStyleAnthropic, http.StatusInternalServerError, "api_error", "streaming unsupported by this server")
		return
	}

	conv := convert.NewAnthropicStream(newID("msg_"), req.Model)

	// message_start must precede any other event, including the heartbeat, and
	// must be sent exactly once no matter who gets there first.
	var startOnce sync.Once
	emitStart := func() {
		startOnce.Do(func() {
			sse.begin()
			for _, e := range conv.Start(0) {
				_ = sse.send(e.Event, e.Data)
			}
		})
	}

	// Anthropic's own stream carries ping events, so clients already expect them.
	stopBeat := sse.keepAlive(r.Context(), firstKeepAlive, keepAliveInterval, func() {
		emitStart()
		_ = sse.send("ping", map[string]any{"type": "ping"})
	})
	defer stopBeat()
	stopWatch := tr.watch(r.Context())
	defer stopWatch()

	// message_start and pings carry no answer, so a route may still be retried.
	var sentContent atomic.Bool

	err := h.streamUpstream(r, modelID, body,
		sentContent.Load,
		func(ev bedrock.StreamEvent) error {
			tr.event(ev)
			if ev.Type == "messageStart" {
				emitStart()
				return nil
			}
			for _, e := range conv.Handle(ev) {
				emitStart()
				sentContent.Store(true)
				if err := sse.send(e.Event, e.Data); err != nil {
					return err
				}
			}
			return nil
		}, tr)

	stopWatch()
	tr.finish(err)

	if err != nil {
		if !sse.written() {
			store.RecordUsage(keyID(key), 0, 0, true)
			upstreamError(w, errStyleAnthropic, r, modelID, err)
			return
		}
		logStreamFailure(r, modelID, err)
		_ = sse.send("error", anthropic.NewErrorEnvelope("api_error", err.Error()))
		store.RecordUsage(keyID(key), 0, 0, !clientHungUp(r, err))
		return
	}

	stopBeat()
	emitStart()
	for _, e := range conv.Finish() {
		_ = sse.send(e.Event, e.Data)
	}
	store.RecordUsage(keyID(key), int64(conv.Usage.InputTokens), int64(conv.Usage.OutputTokens), false)
}
