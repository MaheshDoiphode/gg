package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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

// sseWriter emits server-sent events and flushes each one immediately.
type sseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	wrote   bool
}

func newSSEWriter(w http.ResponseWriter) (*sseWriter, bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, false
	}
	return &sseWriter{w: w, flusher: flusher}, true
}

func (s *sseWriter) begin() {
	s.w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	s.w.Header().Set("Cache-Control", "no-cache")
	s.w.Header().Set("Connection", "keep-alive")
	s.w.Header().Set("X-Accel-Buffering", "no")
	s.w.WriteHeader(http.StatusOK)
	s.flusher.Flush()
}

// send writes one event. An empty name omits the event: line (OpenAI style).
func (s *sseWriter) send(name string, data any) error {
	raw, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if name != "" {
		if _, err := fmt.Fprintf(s.w, "event: %s\n", name); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", raw); err != nil {
		return err
	}
	s.wrote = true
	s.flusher.Flush()
	return nil
}

func (s *sseWriter) raw(line string) {
	fmt.Fprint(s.w, line)
	s.wrote = true
	s.flusher.Flush()
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

	if req.Stream {
		h.streamChat(w, r, key, &req, modelID, body)
		return
	}

	var resp *bedrock.ConverseResponse
	resp, err = h.callUpstream(r, modelID, body)
	if err != nil {
		store.RecordUsage(keyID(key), 0, 0, true)
		upstreamError(w, errStyleOpenAI, r, modelID, err)
		return
	}

	store.RecordUsage(keyID(key), int64(resp.Usage.InputTokens), int64(resp.Usage.OutputTokens), false)
	writeJSON(w, http.StatusOK, convert.ConverseToOpenAI(resp, newID("chatcmpl-"), req.Model, time.Now().Unix()))
}

func (h *Handler) streamChat(w http.ResponseWriter, r *http.Request, key *store.APIKey, req *openai.ChatRequest, modelID string, body *bedrock.ConverseRequest) {
	sse, ok := newSSEWriter(w)
	if !ok {
		writeAPIError(w, errStyleOpenAI, http.StatusInternalServerError, "api_error", "streaming unsupported by this server")
		return
	}

	conv := convert.NewOpenAIStream(newID("chatcmpl-"), req.Model, time.Now().Unix())
	err := h.streamUpstream(r, modelID, body,
		func() bool { return sse.wrote },
		func(ev bedrock.StreamEvent) error {
			for _, chunk := range conv.Handle(ev) {
				if !sse.wrote {
					sse.begin()
				}
				if err := sse.send("", chunk); err != nil {
					return err
				}
			}
			return nil
		})

	if err != nil {
		if !sse.wrote {
			store.RecordUsage(keyID(key), 0, 0, true)
			upstreamError(w, errStyleOpenAI, r, modelID, err)
			return
		}
		logx.Errorf("%s model=%s stream aborted after partial output: %v", r.URL.Path, modelID, err)
		_ = sse.send("", openai.NewError("api_error", "upstream_error", err.Error()))
		sse.raw("data: [DONE]\n\n")
		store.RecordUsage(keyID(key), 0, 0, true)
		return
	}

	if !sse.wrote {
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

	if req.Stream {
		h.streamMessages(w, r, key, &req, modelID, body)
		return
	}

	var resp *bedrock.ConverseResponse
	resp, err = h.callUpstream(r, modelID, body)
	if err != nil {
		store.RecordUsage(keyID(key), 0, 0, true)
		upstreamError(w, errStyleAnthropic, r, modelID, err)
		return
	}

	store.RecordUsage(keyID(key), int64(resp.Usage.InputTokens), int64(resp.Usage.OutputTokens), false)
	writeJSON(w, http.StatusOK, convert.ConverseToAnthropic(resp, newID("msg_"), req.Model))
}

func (h *Handler) streamMessages(w http.ResponseWriter, r *http.Request, key *store.APIKey, req *anthropic.Request, modelID string, body *bedrock.ConverseRequest) {
	sse, ok := newSSEWriter(w)
	if !ok {
		writeAPIError(w, errStyleAnthropic, http.StatusInternalServerError, "api_error", "streaming unsupported by this server")
		return
	}

	conv := convert.NewAnthropicStream(newID("msg_"), req.Model)
	err := h.streamUpstream(r, modelID, body,
		func() bool { return sse.wrote },
		func(ev bedrock.StreamEvent) error {
			var events []convert.SSE
			if ev.Type == "messageStart" {
				events = conv.Start(0)
			} else {
				events = conv.Handle(ev)
			}
			for _, e := range events {
				if !sse.wrote {
					sse.begin()
				}
				if err := sse.send(e.Event, e.Data); err != nil {
					return err
				}
			}
			return nil
		})

	if err != nil {
		if !sse.wrote {
			store.RecordUsage(keyID(key), 0, 0, true)
			upstreamError(w, errStyleAnthropic, r, modelID, err)
			return
		}
		logx.Errorf("%s model=%s stream aborted after partial output: %v", r.URL.Path, modelID, err)
		_ = sse.send("error", anthropic.NewErrorEnvelope("api_error", err.Error()))
		store.RecordUsage(keyID(key), 0, 0, true)
		return
	}

	if !sse.wrote {
		sse.begin()
		for _, e := range conv.Start(0) {
			_ = sse.send(e.Event, e.Data)
		}
	}
	for _, e := range conv.Finish() {
		_ = sse.send(e.Event, e.Data)
	}
	store.RecordUsage(keyID(key), int64(conv.Usage.InputTokens), int64(conv.Usage.OutputTokens), false)
}
