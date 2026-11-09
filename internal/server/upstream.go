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

// needsInferenceProfile reports Converse's rejection when a model requires a
// cross-region inference profile (e.g. moonshotai.kimi-k3.1 -> global.moonshotai.kimi-k3.1).
func needsInferenceProfile(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "on-demand throughput") ||
		strings.Contains(msg, "inference profile that contains this model")
}

// mantleModelMissing reports a 404 from Mantle for a model that only exists on
// Bedrock Runtime Converse (such as Kimi K3.1).
func mantleModelMissing(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "does not exist") ||
		strings.Contains(msg, "is on neither mantle api") ||
		strings.Contains(msg, "model_not_available")
}

// converseBlockedForAccount reports when Converse blocks a model that may
// still be reachable on Mantle (e.g. xai.grok-4.6 under an SCP).
func converseBlockedForAccount(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "not available for this account") ||
		strings.Contains(msg, "operation not allowed")
}

func (h *Handler) invokeConverse(r *http.Request, cred store.Credential, modelID string, body *bedrock.ConverseRequest, tr *trace, attempt int) (*bedrock.ConverseResponse, error) {
	tr.attempt("converse", cred.Region, attempt)
	applyConverseMaxTokens(body)
	out, err := h.client.Converse(r.Context(), cred, modelID, body)
	if needsInferenceProfile(err) && bedrock.StripGeo(modelID) == modelID {
		profID := "global." + modelID
		logx.Debugf("%s model %s requires an inference profile, retrying as %s", tr.id, modelID, profID)
		tr.attempt("converse", cred.Region, attempt)
		out, err = h.client.Converse(r.Context(), cred, profID, body)
	}
	return out, err
}

func (h *Handler) invokeConverseStream(r *http.Request, cred store.Credential, modelID string, body *bedrock.ConverseRequest, emitted func() bool, fn bedrock.StreamFunc, tr *trace, attempt int) error {
	tr.attempt("converse stream", cred.Region, attempt)
	applyConverseMaxTokens(body)
	err := h.client.ConverseStream(r.Context(), cred, modelID, body, fn)
	if needsInferenceProfile(err) && !emitted() && bedrock.StripGeo(modelID) == modelID {
		profID := "global." + modelID
		logx.Debugf("%s model %s requires an inference profile, retrying as %s", tr.id, modelID, profID)
		tr.attempt("converse stream", cred.Region, attempt)
		err = h.client.ConverseStream(r.Context(), cred, profID, body, fn)
	}
	return err
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
			out, callErr := h.invokeConverse(r, cred, modelID, body, tr, attempt)
			if callErr == nil {
				resp = out
				return nil
			}
			if !converseBlockedForAccount(callErr) {
				return callErr
			}
			logx.Debugf("%s converse blocked %s (%v), trying mantle", tr.id, modelID, callErr)
		}

		mantleID := bedrock.StripGeo(modelID)
		if h.routes.get(mantleID) != routeResponses {
			tr.attempt("mantle chat/completions", bedrock.MantleRegionOf(cred), attempt)
			out, callErr := h.client.MantleChat(r.Context(), cred,
				convert.ConverseToOpenAIRequest(mantleID, body, false))
			if callErr == nil {
				h.routes.set(mantleID, routeChat)
				resp = convert.OpenAIResponseToConverse(out)
				return nil
			}
			if !wrongRoute(callErr) {
				if mantleModelMissing(callErr) && h.registry.UpstreamFor(modelID) == bedrock.UpstreamMantle {
					logx.Debugf("%s mantle does not have %s (%v), trying converse", tr.id, mantleID, callErr)
					if convOut, convErr := h.invokeConverse(r, cred, modelID, body, tr, attempt); convErr == nil {
						resp = convOut
						return nil
					}
				}
				return callErr
			}
			logx.Debugf("%s chat/completions rejected %s, trying the responses API: %v", tr.id, mantleID, callErr)
			chatErr = callErr
		}

		// The Responses API can hang when not streamed, so always stream and fold.
		tr.attempt("mantle responses", bedrock.MantleRegionOf(cred), attempt)
		var events []bedrock.StreamEvent
		adapter := convert.NewResponsesStream()
		callErr := h.client.ResponsesStream(r.Context(), cred,
			convert.ConverseToResponsesRequest(mantleID, body),
			func(ev bedrock.ResponsesEvent) error {
				out := adapter.Handle(ev)
				for _, e := range out {
					tr.event(e)
				}
				events = append(events, out...)
				return nil
			})
		if callErr != nil {
			combined := bothRoutesFailed(mantleID, chatErr, callErr)
			if mantleModelMissing(combined) && h.registry.UpstreamFor(modelID) == bedrock.UpstreamMantle {
				logx.Debugf("%s mantle does not have %s (%v), trying converse", tr.id, mantleID, combined)
				if convOut, convErr := h.invokeConverse(r, cred, modelID, body, tr, attempt); convErr == nil {
					resp = convOut
					return nil
				}
			}
			return combined
		}
		if !adapter.Produced() {
			return errEmptyUpstream(mantleID)
		}
		// Only trust the route once it has actually answered.
		if h.routes.get(mantleID) != routeResponses {
			logx.Debugf("model %s answers on the responses API, not chat/completions", mantleID)
		}
		h.routes.set(mantleID, routeResponses)
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

		mantleID := bedrock.StripGeo(modelID)
		tryMantleStream := func() error {
			if h.routes.get(mantleID) == routeResponses {
				tr.attempt("mantle responses stream", bedrock.MantleRegionOf(cred), attempt)
				return h.streamResponses(r, cred, mantleID, body, fn)
			}
			tr.attempt("mantle chat/completions stream", bedrock.MantleRegionOf(cred), attempt)
			adapter := convert.NewMantleStream()
			err := h.client.MantleChatStream(r.Context(), cred,
				convert.ConverseToOpenAIRequest(mantleID, body, true),
				func(chunk *openai.ChatResponse) error {
					for _, ev := range adapter.Handle(chunk) {
						if err := fn(ev); err != nil {
							return err
						}
					}
					return nil
				})
			// A route rejection arrives before any output, so switching is safe.
			if wrongRoute(err) && !emitted() {
				chatErr := err
				logx.Debugf("%s chat/completions rejected %s, trying the responses API: %v", tr.id, mantleID, chatErr)
				tr.attempt("mantle responses stream", bedrock.MantleRegionOf(cred), attempt)
				err = h.streamResponses(r, cred, mantleID, body, fn)
				switch {
				case err == nil:
					if h.routes.get(mantleID) != routeResponses {
						logx.Debugf("model %s answers on the responses API, not chat/completions", mantleID)
					}
					h.routes.set(mantleID, routeResponses)
				default:
					err = bothRoutesFailed(mantleID, chatErr, err)
				}
			}
			return err
		}

		if h.registry.UpstreamFor(modelID) != bedrock.UpstreamMantle {
			callErr = h.invokeConverseStream(r, cred, modelID, body, emitted, fn, tr, attempt)
			if callErr != nil && !emitted() && converseBlockedForAccount(callErr) {
				logx.Debugf("%s converse stream blocked %s (%v), trying mantle", tr.id, modelID, callErr)
				callErr = tryMantleStream()
			}
		} else {
			callErr = tryMantleStream()
			if callErr != nil && !emitted() && mantleModelMissing(callErr) {
				logx.Debugf("%s mantle stream does not have %s (%v), trying converse", tr.id, mantleID, callErr)
				if convErr := h.invokeConverseStream(r, cred, modelID, body, emitted, fn, tr, attempt); convErr == nil {
					callErr = nil
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
