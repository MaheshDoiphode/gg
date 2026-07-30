package convert

import (
	"encoding/json"
	"strings"
	"testing"

	"bedrock-simple/internal/anthropic"
	"bedrock-simple/internal/bedrock"
)

func mustConvertAnthropic(t *testing.T, body string) *bedrock.ConverseRequest {
	t.Helper()
	return mustConvertAnthropicFor(t, body, "anthropic.claude-x")
}

func mustConvertAnthropicFor(t *testing.T, body, model string) *bedrock.ConverseRequest {
	t.Helper()
	var req anthropic.Request
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("decode: %v", err)
	}
	out, err := AnthropicToConverse(&req, model, 4096)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	return out
}

func TestAnthropicSystemAcceptsStringAndArray(t *testing.T) {
	asString := mustConvertAnthropic(t, `{"model":"m","max_tokens":10,"system":"be brief",
		"messages":[{"role":"user","content":"hi"}]}`)
	if len(asString.System) != 1 || *asString.System[0].Text != "be brief" {
		t.Fatalf("string system = %#v", asString.System)
	}

	asArray := mustConvertAnthropic(t, `{"model":"m","max_tokens":10,
		"system":[{"type":"text","text":"a"},{"type":"text","text":"b"}],
		"messages":[{"role":"user","content":"hi"}]}`)
	if len(asArray.System) != 2 {
		t.Fatalf("array system = %#v", asArray.System)
	}
}

func TestAnthropicContentAcceptsStringShorthand(t *testing.T) {
	out := mustConvertAnthropic(t, `{"model":"m","max_tokens":10,
		"messages":[{"role":"user","content":"plain string"}]}`)

	if *out.Messages[0].Content[0].Text != "plain string" {
		t.Fatalf("content = %#v", out.Messages[0].Content[0])
	}
}

func TestAnthropicToolResultBlocks(t *testing.T) {
	out := mustConvertAnthropic(t, `{"model":"m","max_tokens":10,"messages":[
		{"role":"user","content":"go"},
		{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{"a":1}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"42","is_error":true}]}
	]}`)

	res := out.Messages[2].Content[0].ToolResult
	if res == nil || res.ToolUseID != "t1" || res.Status != "error" {
		t.Fatalf("toolResult = %#v", res)
	}
	if *res.Content[0].Text != "42" {
		t.Fatalf("toolResult content = %#v", res.Content[0])
	}
}

func TestAnthropicThinkingClearsSamplingAndRaisesMaxTokens(t *testing.T) {
	out := mustConvertAnthropic(t, `{"model":"m","max_tokens":100,"temperature":0.9,
		"thinking":{"type":"enabled","budget_tokens":8000},
		"messages":[{"role":"user","content":"x"}]}`)

	if out.InferenceConfig.Temperature != nil || out.InferenceConfig.TopP != nil {
		t.Error("sampling params must be cleared while thinking is enabled")
	}
	if *out.InferenceConfig.MaxTokens <= 8000 {
		t.Errorf("maxTokens = %d", *out.InferenceConfig.MaxTokens)
	}
}

// thinking and top_k are Anthropic-only fields; sending them to xai or amazon
// models makes Bedrock reject the whole request.
func TestAnthropicOnlyFieldsAreNotSentToOtherProviders(t *testing.T) {
	out := mustConvertAnthropicFor(t, `{"model":"m","max_tokens":100,"top_k":40,"temperature":0.9,
		"thinking":{"type":"enabled","budget_tokens":8000},
		"messages":[{"role":"user","content":"x"}]}`, "xai.grok-4.3")

	if out.AdditionalModelRequestFields != nil {
		t.Fatalf("non-Anthropic model received provider fields: %#v", out.AdditionalModelRequestFields)
	}
	if out.InferenceConfig.Temperature == nil || *out.InferenceConfig.Temperature != 0.9 {
		t.Error("temperature should be preserved when thinking was not applied")
	}
	if *out.InferenceConfig.MaxTokens != 100 {
		t.Errorf("maxTokens = %d", *out.InferenceConfig.MaxTokens)
	}
}

// Converse does not emit contentBlockStart for plain text, so the Anthropic
// renderer has to open the block itself on the first delta.
func TestAnthropicStreamOpensTextBlockLazily(t *testing.T) {
	s := NewAnthropicStream("msg_1", "model")

	var events []SSE
	events = append(events, s.Start(11)...)
	events = append(events, s.Handle(bedrock.StreamEvent{
		Type: "contentBlockDelta", Delta: &bedrock.ContentBlockDelta{Text: "hi"},
	})...)
	events = append(events, s.Handle(bedrock.StreamEvent{
		Type: "contentBlockDelta", Delta: &bedrock.ContentBlockDelta{Text: " there"},
	})...)
	events = append(events, s.Handle(bedrock.StreamEvent{Type: "contentBlockStop"})...)
	events = append(events, s.Handle(bedrock.StreamEvent{
		Type: "messageStop", StopReason: bedrock.StopEndTurn,
	})...)
	events = append(events, s.Handle(bedrock.StreamEvent{
		Type: "metadata", Usage: &bedrock.TokenUsage{InputTokens: 11, OutputTokens: 2},
	})...)
	events = append(events, s.Finish()...)

	var order []string
	for _, e := range events {
		order = append(order, e.Event)
	}
	want := "message_start,content_block_start,content_block_delta,content_block_delta,content_block_stop,message_delta,message_stop"
	if got := strings.Join(order, ","); got != want {
		t.Fatalf("event order\n got: %s\nwant: %s", got, want)
	}
	if s.Usage.OutputTokens != 2 {
		t.Errorf("usage = %#v", s.Usage)
	}
}

func TestAnthropicStreamToolUseBlock(t *testing.T) {
	s := NewAnthropicStream("msg_1", "model")
	_ = s.Start(0)

	start := s.Handle(bedrock.StreamEvent{
		Type: "contentBlockStart", ContentBlockIndex: 0,
		Start: &bedrock.ContentBlockStart{ToolUse: &bedrock.ToolUseStart{ToolUseID: "t1", Name: "f"}},
	})
	if len(start) != 1 || start[0].Event != "content_block_start" {
		t.Fatalf("start = %#v", start)
	}
	block := start[0].Data.(map[string]any)["content_block"].(map[string]any)
	if block["type"] != "tool_use" || block["id"] != "t1" {
		t.Fatalf("content_block = %#v", block)
	}

	delta := s.Handle(bedrock.StreamEvent{
		Type: "contentBlockDelta", ContentBlockIndex: 0,
		Delta: &bedrock.ContentBlockDelta{ToolUse: &bedrock.ToolUseDelta{Input: `{"a":`}},
	})
	d := delta[0].Data.(map[string]any)["delta"].(map[string]any)
	if d["type"] != "input_json_delta" || d["partial_json"] != `{"a":` {
		t.Fatalf("delta = %#v", d)
	}
}

// A stream that dies before contentBlockStop must still close its blocks,
// otherwise Anthropic clients hang waiting for the terminator.
func TestAnthropicStreamFinishClosesDanglingBlocks(t *testing.T) {
	s := NewAnthropicStream("msg_1", "model")
	_ = s.Start(0)
	_ = s.Handle(bedrock.StreamEvent{Type: "contentBlockDelta", Delta: &bedrock.ContentBlockDelta{Text: "x"}})

	var order []string
	for _, e := range s.Finish() {
		order = append(order, e.Event)
	}
	if got := strings.Join(order, ","); got != "content_block_stop,message_delta,message_stop" {
		t.Fatalf("finish order = %s", got)
	}
}

func TestConverseToAnthropicResponse(t *testing.T) {
	resp := &bedrock.ConverseResponse{StopReason: bedrock.StopMaxTokens}
	resp.Output.Message = &bedrock.Message{Role: "assistant", Content: []bedrock.ContentBlock{
		{ReasoningContent: &bedrock.ReasoningContent{
			ReasoningText: &bedrock.ReasoningText{Text: "hmm", Signature: "sig"}}},
		{Text: bedrock.Ptr("answer")},
	}}
	resp.Usage = bedrock.TokenUsage{InputTokens: 2, OutputTokens: 3}

	out := ConverseToAnthropic(resp, "msg_1", "model")
	if out.Content[0].Type != "thinking" || out.Content[0].Signature != "sig" {
		t.Fatalf("block 0 = %#v", out.Content[0])
	}
	if out.Content[1].Type != "text" || out.Content[1].Text != "answer" {
		t.Fatalf("block 1 = %#v", out.Content[1])
	}
	if out.StopReason != "max_tokens" || out.Usage.InputTokens != 2 {
		t.Fatalf("stop/usage = %q %#v", out.StopReason, out.Usage)
	}
}

func TestEmptyResponseStillHasContentBlock(t *testing.T) {
	out := ConverseToAnthropic(&bedrock.ConverseResponse{}, "msg_1", "model")
	if len(out.Content) != 1 || out.Content[0].Type != "text" {
		t.Fatalf("content = %#v", out.Content)
	}
}
