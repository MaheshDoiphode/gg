package bedrock

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"bedrock-simple/internal/awsauth"
	"bedrock-simple/internal/logx"
	"bedrock-simple/internal/openai"
	"bedrock-simple/internal/store"
)

// Bedrock Mantle is a second, OpenAI-compatible Bedrock endpoint. It carries a
// different catalogue from bedrock-runtime (xai, zai, moonshot, qwen, gpt-oss)
// and some API keys are scoped to it exclusively.

// mantleRoot is the host without an API prefix. Chat Completions lives under
// /v1 and the Responses API under /openai/v1, and the two serve different
// models. Mantle may also live in a different region from Converse.
func mantleRoot(cred store.Credential) string {
	if cred.MantleURL != "" {
		base := strings.TrimSuffix(cred.MantleURL, "/")
		base = strings.TrimSuffix(base, "/openai/v1")
		return strings.TrimSuffix(base, "/v1")
	}
	return "https://bedrock-mantle." + mantleRegion(cred) + ".api.aws"
}

func mantleRegion(cred store.Credential) string {
	if cred.MantleRegion != "" {
		return cred.MantleRegion
	}
	return region(cred)
}

// MantleRegionOf reports the region Mantle calls use, which may differ from the
// credential's Converse region.
func MantleRegionOf(cred store.Credential) string { return mantleRegion(cred) }

func mantleHost(cred store.Credential) string {
	return mantleRoot(cred) + "/v1"
}

func (c *Client) newMantleRequest(ctx context.Context, cred store.Credential, method, url string, body []byte) (*http.Request, error) {
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
		}, region(cred), "bedrock-mantle", time.Now())
	}
	return req, nil
}

// MantleModel is one entry of the Mantle GET /v1/models catalogue.
type MantleModel struct {
	ID           string `json:"id"`
	Object       string `json:"object"`
	OwnedBy      string `json:"owned_by"`
	Status       string `json:"status"`
	StatusReason string `json:"status_reason"`
}

// ListMantleModels returns the Mantle catalogue, including unavailable models
// so the caller can report why one is missing.
func (c *Client) ListMantleModels(ctx context.Context, cred store.Credential) ([]MantleModel, error) {
	req, err := c.newMantleRequest(ctx, cred, http.MethodGet, mantleHost(cred)+"/models", nil)
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
	var out struct {
		Data []MantleModel `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out.Data, nil
}

func (c *Client) postMantleChat(ctx context.Context, cred store.Credential, body *openai.ChatRequest) (*http.Response, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := c.newMantleRequest(ctx, cred, http.MethodPost, mantleHost(cred)+"/chat/completions", raw)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, parseMantleError(resp)
	}
	return resp, nil
}

// MantleChat performs a non-streaming Mantle chat completion.
func (c *Client) MantleChat(ctx context.Context, cred store.Credential, body *openai.ChatRequest) (*openai.ChatResponse, error) {
	body.Stream = false
	body.StreamOptions = nil

	resp, err := c.postMantleChat(ctx, cred, body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var out openai.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode mantle response: %w", err)
	}
	return &out, nil
}

// MantleChatFunc receives each streamed chat completion chunk.
type MantleChatFunc func(chunk *openai.ChatResponse) error

// MantleChatStream performs a streaming Mantle chat completion, invoking fn per
// SSE chunk until the terminating [DONE] sentinel.
func (c *Client) MantleChatStream(ctx context.Context, cred store.Credential, body *openai.ChatRequest, fn MantleChatFunc) error {
	body.Stream = true
	if body.StreamOptions == nil {
		body.StreamOptions = &openai.StreamOptions{IncludeUsage: true}
	}

	resp, err := c.postMantleChat(ctx, cred, body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

	chunks := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		payload, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue
		}
		payload = strings.TrimSpace(payload)
		if payload == "" || payload == "[DONE]" {
			if payload == "[DONE]" {
				logx.Debugf("mantle chat: stream finished after %d chunk(s)", chunks)
				return nil
			}
			continue
		}
		var chunk openai.ChatResponse
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			logx.Debugf("mantle chat: skipped an undecodable chunk: %v", err)
			continue
		}
		chunks++
		if err := fn(&chunk); err != nil {
			return err
		}
	}

	if err := scanner.Err(); err != nil {
		logx.Debugf("mantle chat: stream read failed after %d chunk(s): %v", chunks, err)
		return err
	}
	// Reaching EOF without [DONE] means the answer was cut short, which the
	// scanner reports as a clean end.
	logx.Warnf("mantle chat: stream ended after %d chunk(s) without [DONE]", chunks)
	return &APIError{
		Status:    http.StatusBadGateway,
		ErrorType: "incomplete_stream",
		Message:   "the connection closed before the model finished responding",
	}
}

// parseMantleError reads the OpenAI-shaped error body Mantle returns.
func parseMantleError(resp *http.Response) *APIError {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	e := &APIError{Status: resp.StatusCode}

	var body struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
		Message string `json:"message"`
	}
	if json.Unmarshal(raw, &body) == nil {
		e.Message = firstNonEmpty(body.Error.Message, body.Message)
		e.ErrorType = firstNonEmpty(body.Error.Code, body.Error.Type)
	}
	e.Message = firstNonEmpty(e.Message, strings.TrimSpace(string(raw)), resp.Status)
	return e
}
