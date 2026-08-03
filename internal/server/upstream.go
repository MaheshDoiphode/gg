package server

import (
	"net/http"
	"strings"
	"sync"

	"bedrock-simple/internal/bedrock"
	"bedrock-simple/internal/convert"
	"bedrock-simple/internal/logx"
	"bedrock-simple/internal/openai"
	"bedrock-simple/internal/store"
)

// Mantle exposes two APIs and nothing in its catalogue says which one a model
// accepts: xai.grok-4.3 is rejected by /v1/chat/completions and only answers on
// /openai/v1/responses. routeCache learns that from the first rejection, and
// persists it because learning costs a full prompt upload to the wrong endpoint.
type routeCache struct {
	mu     sync.RWMutex
	byName map[string]string
}

const (
	routeChat      = "chat"
	routeResponses = "responses"
)

func newRouteCache() *routeCache {
	return &routeCache{byName: store.ModelRoutes()}
}

func (r *routeCache) get(model string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.byName[model]
}

func (r *routeCache) set(model, route string) {
	r.mu.Lock()
	known := r.byName[model] == route
	r.byName[model] = route
	r.mu.Unlock()

	if known {
		return
	}
	if err := store.SetModelRoute(model, route); err != nil {
		logx.Debugf("could not persist the %s route for %s: %v", route, model, err)
	}
}

// wrongRoute reports a "model isn't supported on this route" rejection.
func wrongRoute(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "supported on this route") ||
		strings.Contains(msg, "does not support the '/v1/chat/completions'")
}

// applyConverseMaxTokens supplies a cap only for Converse, where some models
// default to a very small output budget. Mantle models are left uncapped so
// they can use their full output length.
func applyConverseMaxTokens(body *bedrock.ConverseRequest) {
	if body.InferenceConfig == nil {
		body.InferenceConfig = &bedrock.InferenceConfig{}
	}
	if body.InferenceConfig.MaxTokens == nil {
		body.InferenceConfig.MaxTokens = bedrock.Ptr(store.DefaultMaxTokens())
	}
}

// callUpstream runs one non-streaming inference against whichever Bedrock API
// serves this model, returning the reply in the hub format.
func (h *Handler) callUpstream(r *http.Request, modelID string, body *bedrock.ConverseRequest, tr *trace) (*bedrock.ConverseResponse, error) {
	var resp *bedrock.ConverseResponse
	var chatErr error
	attempt := 0

	err := h.converse(r, modelID, body, func(cred store.Credential) error {
		attempt++
		if h.registry.UpstreamFor(modelID) != bedrock.UpstreamMantle {
			tr.attempt("converse", cred.Region, attempt)
			applyConverseMaxTokens(body)
			out, callErr := h.client.Converse(r.Context(), cred, modelID, body)
			if callErr != nil {
				return callErr
			}
			resp = out
			return nil
		}

		if h.routes.get(modelID) != routeResponses {
			tr.attempt("mantle chat/completions", bedrock.MantleRegionOf(cred), attempt)
			out, callErr := h.client.MantleChat(r.Context(), cred,
				convert.ConverseToOpenAIRequest(modelID, body, false))
			if callErr == nil {
				h.routes.set(modelID, routeChat)
				resp = convert.OpenAIResponseToConverse(out)
				return nil
			}
			if !wrongRoute(callErr) {
				return callErr
			}
			logx.Debugf("%s chat/completions rejected %s, trying the responses API: %v", tr.id, modelID, callErr)
			chatErr = callErr
		}

		// The Responses API can hang when not streamed, so always stream and fold.
		tr.attempt("mantle responses", bedrock.MantleRegionOf(cred), attempt)
		var events []bedrock.StreamEvent
		adapter := convert.NewResponsesStream()
		callErr := h.client.ResponsesStream(r.Context(), cred,
			convert.ConverseToResponsesRequest(modelID, body),
			func(ev bedrock.ResponsesEvent) error {
				out := adapter.Handle(ev)
				for _, e := range out {
					tr.event(e)
				}
				events = append(events, out...)
				return nil
			})
		if callErr != nil {
			return bothRoutesFailed(modelID, chatErr, callErr)
		}
		if !adapter.Produced() {
			return errEmptyUpstream(modelID)
		}
		// Only trust the route once it has actually answered.
		if h.routes.get(modelID) != routeResponses {
			logx.Debugf("model %s answers on the responses API, not chat/completions", modelID)
		}
		h.routes.set(modelID, routeResponses)
		events = append(events, adapter.Finish()...)
		resp = convert.AggregateToConverse(events)
		return nil
	})
	return resp, err
}

// bothRoutesFailed reports the useful error when a model is on neither Mantle
// API, which usually means the account cannot access it at all.
func bothRoutesFailed(model string, chatErr, respErr error) error {
	if chatErr == nil || !wrongRoute(respErr) {
		return respErr
	}
	return &bedrock.APIError{
		Status:    http.StatusNotFound,
		ErrorType: "model_not_available",
		Message: "model " + model + " is on neither Mantle API: " +
			chatErr.Error() + "; " + respErr.Error(),
	}
}

// streamUpstream runs one streaming inference, emitting hub events to fn.
// emitted reports whether any bytes have already reached the client, which
// makes a failure non-retryable.
func (h *Handler) streamUpstream(r *http.Request, modelID string, body *bedrock.ConverseRequest,
	emitted func() bool, fn bedrock.StreamFunc, tr *trace) error {

	attempt := 0
	return h.converse(r, modelID, body, func(cred store.Credential) error {
		var callErr error
		attempt++

		switch {
		case h.registry.UpstreamFor(modelID) != bedrock.UpstreamMantle:
			tr.attempt("converse stream", cred.Region, attempt)
			applyConverseMaxTokens(body)
			callErr = h.client.ConverseStream(r.Context(), cred, modelID, body, fn)

		case h.routes.get(modelID) == routeResponses:
			tr.attempt("mantle responses stream", bedrock.MantleRegionOf(cred), attempt)
			callErr = h.streamResponses(r, cred, modelID, body, fn)

		default:
			tr.attempt("mantle chat/completions stream", bedrock.MantleRegionOf(cred), attempt)
			adapter := convert.NewMantleStream()
			callErr = h.client.MantleChatStream(r.Context(), cred,
				convert.ConverseToOpenAIRequest(modelID, body, true),
				func(chunk *openai.ChatResponse) error {
					for _, ev := range adapter.Handle(chunk) {
						if err := fn(ev); err != nil {
							return err
						}
					}
					return nil
				})
			// A route rejection arrives before any output, so switching is safe.
			if wrongRoute(callErr) && !emitted() {
				chatErr := callErr
				logx.Debugf("%s chat/completions rejected %s, trying the responses API: %v", tr.id, modelID, chatErr)
				tr.attempt("mantle responses stream", bedrock.MantleRegionOf(cred), attempt)
				callErr = h.streamResponses(r, cred, modelID, body, fn)
				switch {
				case callErr == nil:
					if h.routes.get(modelID) != routeResponses {
						logx.Debugf("model %s answers on the responses API, not chat/completions", modelID)
					}
					h.routes.set(modelID, routeResponses)
				default:
					callErr = bothRoutesFailed(modelID, chatErr, callErr)
				}
			}
		}

		// Once bytes are on the wire a retry would concatenate two answers.
		if callErr != nil && emitted() {
			return &fatalError{callErr}
		}
		return callErr
	})
}

func (h *Handler) streamResponses(r *http.Request, cred store.Credential, modelID string,
	body *bedrock.ConverseRequest, fn bedrock.StreamFunc) error {

	adapter := convert.NewResponsesStream()
	err := h.client.ResponsesStream(r.Context(), cred,
		convert.ConverseToResponsesRequest(modelID, body),
		func(ev bedrock.ResponsesEvent) error {
			for _, out := range adapter.Handle(ev) {
				if err := fn(out); err != nil {
					return err
				}
			}
			return nil
		})
	if err != nil {
		return err
	}
	// An empty answer stalls agent loops without explaining why, so it is
	// reported rather than passed on as a successful empty turn.
	if !adapter.Produced() {
		return errEmptyUpstream(modelID)
	}
	for _, out := range adapter.Finish() {
		if err := fn(out); err != nil {
			return err
		}
	}
	return nil
}

func errEmptyUpstream(model string) error {
	return &bedrock.APIError{
		Status:    http.StatusBadGateway,
		ErrorType: "empty_response",
		Message: "model " + model + " returned no content; reasoning models spend " +
			"max_tokens on thinking before any text, so try a larger budget",
	}
}
