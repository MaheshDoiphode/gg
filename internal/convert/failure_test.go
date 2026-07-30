package convert

import (
	"testing"

	"bedrock-simple/internal/bedrock"
)

// An upstream failure arrives as an ordinary SSE frame. If it is folded into a
// normal stop the client sees a successful empty turn and agent loops stall.
func TestResponsesStreamDoesNotFinishOnFailure(t *testing.T) {
	s := NewResponsesStream()
	events := s.Handle(bedrock.ResponsesEvent{
		Type:     "response.failed",
		Response: &bedrock.ResponsesResult{Error: &bedrock.ResponsesError{Code: "server_error"}},
	})

	for _, e := range events {
		if e.Type == "messageStop" {
			t.Fatal("a failed response must not be reported as a normal stop")
		}
	}
	if s.Produced() {
		t.Error("no content was produced")
	}
}

func TestResponsesStreamProduced(t *testing.T) {
	text := NewResponsesStream()
	if text.Produced() {
		t.Error("a fresh stream has produced nothing")
	}
	text.Handle(bedrock.ResponsesEvent{Type: "response.output_text.delta", Delta: "hi"})
	if !text.Produced() {
		t.Error("text output should count as content")
	}

	// A tool call with no text is still a real answer.
	tool := NewResponsesStream()
	tool.Handle(bedrock.ResponsesEvent{Type: "response.output_item.added",
		Item: &bedrock.ResponsesItem{Type: "function_call", ID: "fc_1", CallID: "c1", Name: "f"}})
	if !tool.Produced() {
		t.Error("a tool call should count as content")
	}

	// Reasoning alone is not an answer: Grok emits reasoning items with no
	// visible output when it fails.
	reasoning := NewResponsesStream()
	reasoning.Handle(bedrock.ResponsesEvent{Type: "response.created"})
	reasoning.Handle(bedrock.ResponsesEvent{Type: "response.in_progress"})
	if reasoning.Produced() {
		t.Error("lifecycle events alone are not content")
	}
}
