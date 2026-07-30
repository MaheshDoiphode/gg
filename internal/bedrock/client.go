// Package bedrock is a minimal AWS Bedrock client built on net/http alone.
// It speaks the Converse / ConverseStream runtime API and the control-plane
// model-listing API, authenticating with either a Bedrock API key (bearer) or
// hand-rolled SigV4.
package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"bedrock-simple/internal/awsauth"
	"bedrock-simple/internal/store"
)

// Client talks to Bedrock.
type Client struct {
	http *http.Client
	next uint64
}

// New builds a client. The timeout is generous because a long non-streaming
// generation can legitimately take minutes.
func New() *Client {
	return &Client{
		http: &http.Client{
			Timeout: 30 * time.Minute,
			Transport: &http.Transport{
				MaxIdleConnsPerHost:   32,
				IdleConnTimeout:       90 * time.Second,
				TLSHandshakeTimeout:   15 * time.Second,
				ResponseHeaderTimeout: 5 * time.Minute,
				ForceAttemptHTTP2:     true,
			},
		},
	}
}

// APIError is a non-2xx response from Bedrock.
type APIError struct {
	Status    int
	ErrorType string
	Message   string
}

func (e *APIError) Error() string {
	if e.ErrorType != "" {
		return fmt.Sprintf("bedrock %d %s: %s", e.Status, e.ErrorType, e.Message)
	}
	return fmt.Sprintf("bedrock %d: %s", e.Status, e.Message)
}

// Retryable reports whether retrying on a different credential could help.
func (e *APIError) Retryable() bool {
	switch e.Status {
	case http.StatusTooManyRequests,
		http.StatusRequestTimeout,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout,
		424: // ModelErrorException
		return true
	}
	return false
}

// ErrNoCredentials means nothing usable is configured.
var ErrNoCredentials = errors.New("no enabled Bedrock credentials configured")

// PickCredential returns the next usable credential, round-robin, skipping
// disabled ones, those in cooldown, and any id in exclude.
func (c *Client) PickCredential(exclude map[string]bool) (store.Credential, error) {
	creds := store.Credentials()
	usable := make([]store.Credential, 0, len(creds))
	now := time.Now().Unix()
	for _, cr := range creds {
		if cr.Enabled && !exclude[cr.ID] && cr.CooldownUntil <= now {
			usable = append(usable, cr)
		}
	}
	if len(usable) == 0 {
		// Everything is cooling down: prefer a stale credential over failing.
		for _, cr := range creds {
			if cr.Enabled && !exclude[cr.ID] {
				usable = append(usable, cr)
			}
		}
	}
	if len(usable) == 0 {
		return store.Credential{}, ErrNoCredentials
	}
	i := atomic.AddUint64(&c.next, 1) - 1
	return usable[i%uint64(len(usable))], nil
}

func region(cred store.Credential) string {
	if cred.Region != "" {
		return cred.Region
	}
	return "us-east-1"
}

// runtimeHost is the data plane; controlHost is the model-listing plane.
func runtimeHost(cred store.Credential) string {
	if cred.EndpointURL != "" {
		return strings.TrimSuffix(cred.EndpointURL, "/")
	}
	return "https://bedrock-runtime." + region(cred) + ".amazonaws.com"
}

func controlHost(cred store.Credential) string {
	if cred.EndpointURL != "" {
		base := strings.TrimSuffix(cred.EndpointURL, "/")
		return strings.Replace(base, "://bedrock-runtime.", "://bedrock.", 1)
	}
	return "https://bedrock." + region(cred) + ".amazonaws.com"
}

func (c *Client) newRequest(ctx context.Context, cred store.Credential, method, url string, body []byte) (*http.Request, error) {
	var rdr io.Reader
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, rdr)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
		req.ContentLength = int64(len(body))
	}

	switch cred.AuthMode {
	case store.AuthBearer:
		if cred.BearerKey == "" {
			return nil, fmt.Errorf("credential %q is in bearer mode but has no key", cred.Name)
		}
		req.Header.Set("Authorization", "Bearer "+cred.BearerKey)
	default:
		if cred.AccessKey == "" || cred.SecretKey == "" {
			return nil, fmt.Errorf("credential %q is in sigv4 mode but has no access/secret key", cred.Name)
		}
		awsauth.Sign(req, body, awsauth.Creds{
			AccessKey:    cred.AccessKey,
			SecretKey:    cred.SecretKey,
			SessionToken: cred.SessionToken,
		}, region(cred), "bedrock", time.Now())
	}
	return req, nil
}

func parseError(resp *http.Response) *APIError {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	e := &APIError{Status: resp.StatusCode, ErrorType: resp.Header.Get("x-amzn-ErrorType")}

	var body struct {
		Message string `json:"message"`
		Msg     string `json:"Message"`
		Type    string `json:"__type"`
	}
	if json.Unmarshal(raw, &body) == nil {
		e.Message = firstNonEmpty(body.Message, body.Msg)
		if e.ErrorType == "" {
			e.ErrorType = body.Type
		}
	}
	e.Message = firstNonEmpty(e.Message, strings.TrimSpace(string(raw)), resp.Status)
	// The header looks like "ThrottlingException:http://internal.amazon...".
	if i := strings.IndexAny(e.ErrorType, ":#"); i > 0 {
		e.ErrorType = e.ErrorType[:i]
	}
	return e
}

// Converse performs a non-streaming Converse call.
func (c *Client) Converse(ctx context.Context, cred store.Credential, modelID string, body *ConverseRequest) (*ConverseResponse, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	url := runtimeHost(cred) + "/model/" + awsauth.EscapePathSegment(modelID) + "/converse"

	req, err := c.newRequest(ctx, cred, http.MethodPost, url, raw)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, parseError(resp)
	}
	var out ConverseResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode converse response: %w", err)
	}
	return &out, nil
}

// StreamFunc receives each decoded ConverseStream event.
type StreamFunc func(ev StreamEvent) error

// ConverseStream performs a streaming Converse call, invoking fn per event.
// An error returned by fn aborts the stream.
func (c *Client) ConverseStream(ctx context.Context, cred store.Credential, modelID string, body *ConverseRequest, fn StreamFunc) error {
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}
	url := runtimeHost(cred) + "/model/" + awsauth.EscapePathSegment(modelID) + "/converse-stream"

	req, err := c.newRequest(ctx, cred, http.MethodPost, url, raw)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return parseError(resp)
	}

	stream := awsauth.NewEventStream(resp.Body)
	for {
		frame, err := stream.Next()
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return nil
		}
		if err != nil {
			return err
		}

		if mt := frame.MessageType(); mt == "exception" || mt == "error" {
			return &APIError{
				Status:    exceptionStatus(frame.ExceptionType()),
				ErrorType: firstNonEmpty(frame.ExceptionType(), frame.Headers[":error-code"]),
				Message:   extractMessage(frame.Payload),
			}
		}

		ev := StreamEvent{Type: frame.EventType()}
		if ev.Type == "" {
			continue
		}
		if err := json.Unmarshal(frame.Payload, &ev); err != nil {
			continue
		}
		if err := fn(ev); err != nil {
			return err
		}
	}
}

// exceptionStatus maps a mid-stream exception name back to an HTTP status so
// retry logic can treat it like a failed request.
func exceptionStatus(name string) int {
	switch {
	case strings.HasPrefix(name, "throttling"), strings.HasPrefix(name, "Throttling"):
		return http.StatusTooManyRequests
	case strings.HasPrefix(name, "validation"), strings.HasPrefix(name, "Validation"):
		return http.StatusBadRequest
	case strings.HasPrefix(name, "serviceUnavailable"), strings.HasPrefix(name, "ServiceUnavailable"):
		return http.StatusServiceUnavailable
	case strings.HasPrefix(name, "modelStreamError"), strings.HasPrefix(name, "ModelStreamError"):
		return 424
	default:
		return http.StatusBadGateway
	}
}

// getJSON performs a signed control-plane GET and decodes the body into out.
func (c *Client) getJSON(ctx context.Context, cred store.Credential, url string, out any) error {
	req, err := c.newRequest(ctx, cred, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return parseError(resp)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// FoundationModel is one entry of ListFoundationModels.
type FoundationModel struct {
	ModelID                 string   `json:"modelId"`
	ModelARN                string   `json:"modelArn"`
	ModelName               string   `json:"modelName"`
	ProviderName            string   `json:"providerName"`
	InputModalities         []string `json:"inputModalities"`
	OutputModalities        []string `json:"outputModalities"`
	InferenceTypesSupported []string `json:"inferenceTypesSupported"`
	ResponseStreaming       bool     `json:"responseStreamingSupported"`
	ModelLifecycle          struct {
		Status string `json:"status"`
	} `json:"modelLifecycle"`
}

// ListFoundationModels calls GET /foundation-models on the control plane.
func (c *Client) ListFoundationModels(ctx context.Context, cred store.Credential) ([]FoundationModel, error) {
	var out struct {
		ModelSummaries []FoundationModel `json:"modelSummaries"`
	}
	url := controlHost(cred) + "/foundation-models?byOutputModality=TEXT"
	if err := c.getJSON(ctx, cred, url, &out); err != nil {
		return nil, err
	}
	return out.ModelSummaries, nil
}

// InferenceProfile is one entry of ListInferenceProfiles.
type InferenceProfile struct {
	InferenceProfileID   string `json:"inferenceProfileId"`
	InferenceProfileARN  string `json:"inferenceProfileArn"`
	InferenceProfileName string `json:"inferenceProfileName"`
	Description          string `json:"description"`
	Status               string `json:"status"`
	Type                 string `json:"type"`
	Models               []struct {
		ModelARN string `json:"modelArn"`
	} `json:"models"`
}

// ListInferenceProfiles calls GET /inference-profiles, following pagination.
// System-defined profiles are the us./eu./apac./global. cross-region ids.
func (c *Client) ListInferenceProfiles(ctx context.Context, cred store.Credential) ([]InferenceProfile, error) {
	var all []InferenceProfile
	next := ""
	for page := 0; page < 20; page++ {
		url := controlHost(cred) + "/inference-profiles?maxResults=1000&type=SYSTEM_DEFINED"
		if next != "" {
			url += "&nextToken=" + awsauth.EscapePathSegment(next)
		}
		var out struct {
			Summaries []InferenceProfile `json:"inferenceProfileSummaries"`
			NextToken string             `json:"nextToken"`
		}
		if err := c.getJSON(ctx, cred, url, &out); err != nil {
			return all, err
		}
		all = append(all, out.Summaries...)
		if out.NextToken == "" {
			return all, nil
		}
		next = out.NextToken
	}
	return all, nil
}

func extractMessage(payload []byte) string {
	var body struct {
		Message string `json:"message"`
		Msg     string `json:"Message"`
	}
	if json.Unmarshal(payload, &body) == nil {
		if m := firstNonEmpty(body.Message, body.Msg); m != "" {
			return m
		}
	}
	return strings.TrimSpace(string(payload))
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
