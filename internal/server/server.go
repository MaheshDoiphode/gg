// Package server exposes the OpenAI- and Anthropic-compatible HTTP surface.
package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"bedrock-simple/internal/bedrock"
	"bedrock-simple/internal/logx"
	"bedrock-simple/internal/store"
)

// Handler routes every request. It implements http.Handler directly; there is
// no router dependency.
type Handler struct {
	client   *bedrock.Client
	registry *bedrock.Registry
	routes   *routeCache
}

// New builds the handler.
func New(c *bedrock.Client, r *bedrock.Registry) *Handler {
	return &Handler{client: c, registry: r, routes: newRouteCache()}
}

const maxCredentialAttempts = 3

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers",
		"Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta, "+
			"openai-organization, openai-beta, x-stainless-os, x-stainless-lang, "+
			"x-stainless-package-version, x-stainless-runtime, x-stainless-runtime-version, "+
			"x-stainless-arch, x-stainless-retry-count, x-stainless-timeout")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	path := strings.TrimSuffix(r.URL.Path, "/")
	switch path {
	case "/v1/chat/completions", "/chat/completions", "/openai/v1/chat/completions":
		h.requirePost(w, r, h.handleChatCompletions, errStyleOpenAI)
	case "/v1/messages", "/messages", "/anthropic/v1/messages":
		h.requirePost(w, r, h.handleMessages, errStyleAnthropic)
	case "/v1/messages/count_tokens", "/messages/count_tokens":
		h.requirePost(w, r, h.handleCountTokens, errStyleAnthropic)
	case "/v1/models", "/models", "/openai/v1/models":
		h.handleModels(w, r)
	case "/health", "":
		h.handleHealth(w, r)
	default:
		http.NotFound(w, r)
	}
}

type errStyle int

const (
	errStyleOpenAI errStyle = iota
	errStyleAnthropic
)

func (h *Handler) requirePost(w http.ResponseWriter, r *http.Request, fn func(http.ResponseWriter, *http.Request, *store.APIKey), style errStyle) {
	if r.Method != http.MethodPost {
		writeAPIError(w, style, http.StatusMethodNotAllowed, "invalid_request_error", "method not allowed")
		return
	}
	key, err := authenticate(r)
	if err != nil {
		var ae *authError
		errors.As(err, &ae)
		logx.Warnf("%s %s rejected: %s", r.Method, r.URL.Path, ae.message)
		writeAPIError(w, style, ae.status, ae.kind, ae.message)
		return
	}
	fn(w, r, key)
}

// ---------------------------------------------------------------------- auth

type authError struct {
	status  int
	kind    string
	message string
}

func (e *authError) Error() string { return e.message }

func authenticate(r *http.Request) (*store.APIKey, error) {
	if !store.RequireAPIKey() {
		return nil, nil
	}
	provided := strings.TrimSpace(r.Header.Get("x-api-key"))
	if provided == "" {
		if v := r.Header.Get("Authorization"); strings.HasPrefix(v, "Bearer ") {
			provided = strings.TrimSpace(strings.TrimPrefix(v, "Bearer "))
		}
	}
	if provided == "" {
		return nil, &authError{http.StatusUnauthorized, "authentication_error", "missing API key"}
	}
	if !store.HasAPIKeys() {
		// Auth is on but nothing is configured: fail closed rather than open.
		return nil, &authError{http.StatusUnauthorized, "authentication_error", "no API keys are configured on this proxy"}
	}
	key := store.FindAPIKey(provided)
	if key == nil {
		return nil, &authError{http.StatusUnauthorized, "authentication_error", "invalid API key"}
	}
	if !key.Enabled {
		return nil, &authError{http.StatusUnauthorized, "authentication_error", "API key is disabled"}
	}
	if store.OverLimit(*key) {
		return nil, &authError{http.StatusTooManyRequests, "rate_limit_error", "API key token limit exceeded"}
	}
	return key, nil
}

// ------------------------------------------------------------------ failover

// fatalError marks a failure that must not be retried on another credential,
// typically because response bytes have already been flushed to the client.
type fatalError struct{ err error }

func (e *fatalError) Error() string { return e.err.Error() }
func (e *fatalError) Unwrap() error { return e.err }

// withCredential runs fn against usable credentials, retrying transient
// failures on the next one.
func (h *Handler) withCredential(fn func(cred store.Credential) error) error {
	excluded := map[string]bool{}
	var lastErr error

	for attempt := 0; attempt < maxCredentialAttempts; attempt++ {
		cred, err := h.client.PickCredential(excluded)
		if err != nil {
			if lastErr != nil {
				return lastErr
			}
			return err
		}

		err = fn(cred)
		store.RecordCredentialResult(cred.ID, err, cooldownFor(err))
		if err == nil {
			return nil
		}
		lastErr = err

		var fatal *fatalError
		if errors.As(err, &fatal) {
			return err
		}
		var apiErr *bedrock.APIError
		if errors.As(err, &apiErr) && apiErr.Retryable() {
			logx.Warnf("credential %q failed (%v), trying another", cred.Name, err)
			excluded[cred.ID] = true
			continue
		}
		return err
	}
	return lastErr
}

// converse runs fn with credential failover, and retries with a field removed
// when the model rejects one that Converse otherwise allows. body is mutated in
// place, so a retry sends the trimmed request.
func (h *Handler) converse(r *http.Request, model string, body *bedrock.ConverseRequest, fn func(cred store.Credential) error) error {
	var err error
	for attempt := 0; attempt <= maxParamStripRetries; attempt++ {
		err = h.withCredential(fn)
		if err == nil {
			return nil
		}
		dropped := dropRejectedParam(body, err)
		if dropped == "" {
			return err
		}
		logx.Warnf("%s model=%s rejected %q, retrying without it", r.URL.Path, model, dropped)
	}
	return err
}

func cooldownFor(err error) time.Duration {
	var apiErr *bedrock.APIError
	if !errors.As(err, &apiErr) {
		return 0
	}
	switch apiErr.Status {
	case http.StatusTooManyRequests:
		return 60 * time.Second
	case http.StatusUnauthorized, http.StatusForbidden:
		return 5 * time.Minute
	case http.StatusServiceUnavailable, http.StatusInternalServerError, http.StatusBadGateway:
		return 30 * time.Second
	}
	return 0
}

// ------------------------------------------------------------------- replies

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func writeAPIError(w http.ResponseWriter, style errStyle, status int, kind, msg string) {
	if style == errStyleAnthropic {
		writeJSON(w, status, map[string]any{
			"type":  "error",
			"error": map[string]string{"type": kind, "message": msg},
		})
		return
	}
	writeJSON(w, status, map[string]any{
		"error": map[string]string{"type": kind, "message": msg},
	})
}

// upstreamError renders a Bedrock failure with its original status where
// possible, and is the one place request failures get logged.
func upstreamError(w http.ResponseWriter, style errStyle, r *http.Request, model string, err error) {
	status := http.StatusBadGateway
	kind := "api_error"
	msg := err.Error()

	var apiErr *bedrock.APIError
	switch {
	case errors.As(err, &apiErr):
		status = apiErr.Status
		msg = apiErr.Message
		switch status {
		case http.StatusBadRequest:
			kind = "invalid_request_error"
		case http.StatusUnauthorized, http.StatusForbidden:
			kind = "permission_error"
		case http.StatusNotFound:
			kind = "not_found_error"
		case http.StatusTooManyRequests:
			kind = "rate_limit_error"
		}
	case errors.Is(err, bedrock.ErrNoCredentials):
		status = http.StatusServiceUnavailable
		kind = "api_error"
		msg = "no Bedrock credentials are configured; add one under \"credentials\" in " + store.Path()
	case errors.Is(err, context.Canceled):
		return // client hung up
	}

	logx.Errorf("%s model=%s -> %d %s", r.URL.Path, model, status, msg)
	writeAPIError(w, style, status, kind, msg)
}

// ------------------------------------------------------------------ endpoints

func (h *Handler) handleHealth(w http.ResponseWriter, _ *http.Request) {
	count, updated, lastErr := h.registry.Status()
	body := map[string]any{
		"status":      "ok",
		"models":      count,
		"credentials": len(store.Credentials()),
	}
	if !updated.IsZero() {
		body["modelsUpdatedAt"] = updated.UTC().Format(time.RFC3339)
	}
	if lastErr != nil {
		body["modelDiscoveryError"] = lastErr.Error()
	}
	writeJSON(w, http.StatusOK, body)
}

func (h *Handler) handleModels(w http.ResponseWriter, _ *http.Request) {
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	now := time.Now().Unix()
	seen := map[string]bool{}
	data := []model{}

	add := func(id, owner string) {
		if id == "" || seen[id] {
			return
		}
		seen[id] = true
		data = append(data, model{ID: id, Object: "model", Created: now, OwnedBy: owner})
	}
	for _, e := range h.registry.Entries() {
		add(e.ID, e.Provider)
		for _, a := range e.Aliases {
			add(a, e.Provider)
		}
	}
	for alias := range store.ModelMap() {
		add(alias, "custom")
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func newID(prefix string) string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		return prefix + fmt.Sprint(time.Now().UnixNano())
	}
	return prefix + hex.EncodeToString(b)
}
