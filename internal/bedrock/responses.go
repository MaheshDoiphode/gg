package bedrock

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"strings"

	"bedrock-simple/internal/logx"
	"bedrock-simple/internal/store"
)

// The Responses API lives at /openai/v1/responses on the Mantle host. Some
// models are reachable only here: xai.grok-4.3 rejects /v1/chat/completions
// with "isn't supported on this route", and hangs forever on /v1/responses.

// ResponsesRequest is the body of POST /openai/v1/responses.
type ResponsesRequest struct {
	Model           string              `json:"model"`
	Input           []ResponsesInput    `json:"input"`
	Instructions    string              `json:"instructions,omitempty"`
	MaxOutputTokens *int                `json:"max_output_tokens,omitempty"`
	Temperature     *float64            `json:"temperature,omitempty"`
	TopP            *float64            `json:"top_p,omitempty"`
	Stream          bool                `json:"stream"`
	Reasoning       *ResponsesReasoning `json:"reasoning,omitempty"`
	Tools           []ResponsesTool     `json:"tools,omitempty"`
	ToolChoice      json.RawMessage     `json:"tool_choice,omitempty"`
}

// ResponsesReasoning controls how much the model thinks before answering.
type ResponsesReasoning struct {
	Effort string `json:"effort,omitempty"`
}

// ResponsesInput is one input item: a message, a function call, or its output.
type ResponsesInput struct {
	Type      string             `json:"type,omitempty"`
	Role      string             `json:"role,omitempty"`
	Content   []ResponsesContent `json:"content,omitempty"`
	CallID    string             `json:"call_id,omitempty"`
	Name      string             `json:"name,omitempty"`
	Arguments string             `json:"arguments,omitempty"`
	Output    string             `json:"output,omitempty"`
}

// ResponsesContent is one part of a message. The Responses API rejects Chat
// Completions part types, so these must be input_text / input_image / output_text.
type ResponsesContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

// ResponsesTool is a function tool, flattened rather than nested.
type ResponsesTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ResponsesEvent is one decoded SSE frame from the Responses API.
type ResponsesEvent struct {
	Type        string           `json:"type"`
	Delta       string           `json:"delta,omitempty"`
	ItemID      string           `json:"item_id,omitempty"`
	OutputIndex int              `json:"output_index,omitempty"`
	Item        *ResponsesItem   `json:"item,omitempty"`
	Response    *ResponsesResult `json:"response,omitempty"`
}

// ResponsesItem is an output item such as a function call or a message.
type ResponsesItem struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ResponsesResult is the response envelope carried by lifecycle events.
type ResponsesResult struct {
	ID                string               `json:"id"`
	Status            string               `json:"status"`
	Usage             *ResponsesUsage      `json:"usage"`
	Error             *ResponsesError      `json:"error"`
	IncompleteDetails *ResponsesIncomplete `json:"incomplete_details"`
}

// ResponsesIncomplete says why the model stopped before finishing.
type ResponsesIncomplete struct {
	Reason string `json:"reason"`
}

// ResponsesError is the failure carried by a response.failed event.
type ResponsesError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ResponsesUsage is Responses API token accounting.
type ResponsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// ResponsesFunc receives each decoded Responses event.
type ResponsesFunc func(ev ResponsesEvent) error

// ResponsesStream calls the Responses API and invokes fn per event. It always
// streams: the non-streaming form of this route can hang indefinitely.
func (c *Client) ResponsesStream(ctx context.Context, cred store.Credential, body *ResponsesRequest, fn ResponsesFunc) error {
	body.Stream = true
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}

	url := mantleRoot(cred) + "/openai/v1/responses"
	req, err := c.newMantleRequest(ctx, cred, http.MethodPost, url, raw)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return parseMantleError(resp)
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

	frames, terminal := 0, ""
	for scanner.Scan() {
		payload, ok := strings.CutPrefix(strings.TrimSpace(scanner.Text()), "data:")
		if !ok {
			continue
		}
		payload = strings.TrimSpace(payload)
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var ev ResponsesEvent
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			logx.Debugf("responses: skipped an undecodable frame: %v", err)
			continue
		}
		frames++
		// A failure arrives as an ordinary SSE frame, not an HTTP error, so it
		// must be raised here or it silently becomes an empty answer.
		if ev.Type == "response.failed" {
			logx.Debugf("responses: upstream reported failure after %d frame(s)", frames)
			return responseFailure(ev)
		}
		if ev.Type == "response.completed" || ev.Type == "response.incomplete" {
			terminal = ev.Type
			if ev.Type == "response.incomplete" {
				logx.Warnf("responses: model stopped early (%s)", incompleteReason(ev))
			}
		}
		if err := fn(ev); err != nil {
			return err
		}
	}

	if err := scanner.Err(); err != nil {
		logx.Debugf("responses: stream read failed after %d frame(s): %v", frames, err)
		return err
	}
	// EOF is not an error to the scanner, so a connection cut mid-answer would
	// otherwise be indistinguishable from a finished one.
	if terminal == "" {
		logx.Warnf("responses: stream ended after %d frame(s) without a completion event", frames)
		return &APIError{
			Status:    http.StatusBadGateway,
			ErrorType: "incomplete_stream",
			Message:   "the connection closed before the model finished responding",
		}
	}
	logx.Debugf("responses: stream finished with %s after %d frame(s)", terminal, frames)
	return nil
}

// incompleteReason explains a response.incomplete event, which is usually the
// output budget running out.
func incompleteReason(ev ResponsesEvent) string {
	if ev.Response != nil && ev.Response.IncompleteDetails != nil && ev.Response.IncompleteDetails.Reason != "" {
		return ev.Response.IncompleteDetails.Reason
	}
	return "reason not given"
}

func responseFailure(ev ResponsesEvent) error {
	e := &APIError{Status: http.StatusBadGateway, ErrorType: "server_error",
		Message: "the model failed to generate a response"}
	if ev.Response != nil && ev.Response.Error != nil {
		if ev.Response.Error.Code != "" {
			e.ErrorType = ev.Response.Error.Code
		}
		if ev.Response.Error.Message != "" {
			e.Message = ev.Response.Error.Message
		}
	}
	return e
}
