package convert

import (
	"encoding/json"
	"strings"
	"testing"

	"bedrock-simple/internal/bedrock"
	"bedrock-simple/internal/openai"
)

func mustConvert(t *testing.T, body string) *bedrock.ConverseRequest {
	t.Helper()
	var req openai.ChatRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("decode: %v", err)
	}
	out, err := OpenAIToConverse(&req, "anthropic.claude-x", 4096)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	return out
}

func TestSystemMessagesBecomeSystemBlocks(t *testing.T) {
	out := mustConvert(t, `{
		"model":"m",
		"messages":[
			{"role":"system","content":"be brief"},
			{"role":"user","content":"hi"}
		]}`)

	if len(out.System) != 1 || *out.System[0].Text != "be brief" {
		t.Fatalf("system = %#v", out.System)
	}
	if len(out.Messages) != 1 || out.Messages[0].Role != "user" {
		t.Fatalf("messages = %#v", out.Messages)
	}
	if *out.Messages[0].Content[0].Text != "hi" {
		t.Fatalf("content = %#v", out.Messages[0].Content[0])
	}
}

// Converse rejects two consecutive turns with the same role, so adjacent
// same-role messages have to be merged.
func TestConsecutiveSameRoleMessagesMerge(t *testing.T) {
	out := mustConvert(t, `{
		"model":"m",
		"messages":[
			{"role":"user","content":"one"},
			{"role":"user","content":"two"},
			{"role":"assistant","content":"ok"},
			{"role":"user","content":"three"}
		]}`)

	if len(out.Messages) != 3 {
		t.Fatalf("expected 3 turns, got %d: %#v", len(out.Messages), out.Messages)
	}
	if len(out.Messages[0].Content) != 2 {
		t.Fatalf("first turn should hold both user blocks: %#v", out.Messages[0].Content)
	}
}

func TestToolCallsAndResultsRoundTrip(t *testing.T) {
	out := mustConvert(t, `{
		"model":"m",
		"messages":[
			{"role":"user","content":"weather?"},
			{"role":"assistant","tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Pune\"}"}}
			]},
			{"role":"tool","tool_call_id":"call_1","content":"31C"}
		],
		"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]
	}`)

	assistant := out.Messages[1]
	if assistant.Role != "assistant" || assistant.Content[0].ToolUse == nil {
		t.Fatalf("assistant turn = %#v", assistant)
	}
	if got := assistant.Content[0].ToolUse.Name; got != "get_weather" {
		t.Errorf("tool name = %q", got)
	}

	result := out.Messages[2]
	if result.Role != "user" || result.Content[0].ToolResult == nil {
		t.Fatalf("tool result should ride on a user turn: %#v", result)
	}
	if got := result.Content[0].ToolResult.ToolUseID; got != "call_1" {
		t.Errorf("toolUseId = %q", got)
	}
	if out.ToolConfig == nil || len(out.ToolConfig.Tools) != 1 {
		t.Fatalf("toolConfig = %#v", out.ToolConfig)
	}
}

func TestMalformedToolArgumentsFallBackToEmptyObject(t *testing.T) {
	out := mustConvert(t, `{
		"model":"m",
		"messages":[
			{"role":"user","content":"x"},
			{"role":"assistant","tool_calls":[
				{"id":"c","type":"function","function":{"name":"f","arguments":""}}
			]}
		]}`)

	if got := string(out.Messages[1].Content[0].ToolUse.Input); got != "{}" {
		t.Fatalf("input = %s", got)
	}
}

func TestInlineImageBecomesImageBlock(t *testing.T) {
	out := mustConvert(t, `{
		"model":"m",
		"messages":[{"role":"user","content":[
			{"type":"text","text":"what is this"},
			{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}
		]}]}`)

	blocks := out.Messages[0].Content
	if len(blocks) != 2 || blocks[1].Image == nil {
		t.Fatalf("blocks = %#v", blocks)
	}
	if blocks[1].Image.Format != "png" || blocks[1].Image.Source.Bytes != "iVBORw0KGgo=" {
		t.Fatalf("image = %#v", blocks[1].Image)
	}
}

func TestRemoteImageURLIsReportedNotDropped(t *testing.T) {
	out := mustConvert(t, `{
		"model":"m",
		"messages":[{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}
		]}]}`)

	blocks := out.Messages[0].Content
	if len(blocks) != 1 || blocks[0].Text == nil || !strings.Contains(*blocks[0].Text, "image omitted") {
		t.Fatalf("blocks = %#v", blocks)
	}
}

func TestMaxCompletionTokensWins(t *testing.T) {
	out := mustConvert(t, `{"model":"m","max_tokens":10,"max_completion_tokens":99,
		"messages":[{"role":"user","content":"x"}]}`)

	if *out.InferenceConfig.MaxTokens != 99 {
		t.Fatalf("maxTokens = %d", *out.InferenceConfig.MaxTokens)
	}
}

func TestReasoningEffortEnablesThinkingAndClearsTemperature(t *testing.T) {
	out := mustConvert(t, `{"model":"m","reasoning_effort":"high","temperature":0.7,"max_tokens":100,
		"messages":[{"role":"user","content":"x"}]}`)

	thinking, ok := out.AdditionalModelRequestFields["thinking"].(map[string]any)
	if !ok || thinking["budget_tokens"] != 16384 {
		t.Fatalf("thinking = %#v", out.AdditionalModelRequestFields)
	}
	if out.InferenceConfig.Temperature != nil {
		t.Error("temperature must be dropped while thinking is enabled")
	}
	if *out.InferenceConfig.MaxTokens <= 16384 {
		t.Errorf("maxTokens must exceed the thinking budget, got %d", *out.InferenceConfig.MaxTokens)
	}
}

func TestReasoningEffortIgnoredForNonAnthropicModels(t *testing.T) {
	var req openai.ChatRequest
	_ = json.Unmarshal([]byte(`{"model":"m","reasoning_effort":"high",
		"messages":[{"role":"user","content":"x"}]}`), &req)

	out, err := OpenAIToConverse(&req, "meta.llama3-70b-instruct-v1:0", 4096)
	if err != nil {
		t.Fatal(err)
	}
	if out.AdditionalModelRequestFields != nil {
		t.Fatalf("thinking must not be sent to non-Anthropic models: %#v", out.AdditionalModelRequestFields)
	}
}

func TestToolChoiceMapping(t *testing.T) {
	cases := map[string]func(*bedrock.ToolChoice) bool{
		`"auto"`:     func(c *bedrock.ToolChoice) bool { return c != nil && c.Auto != nil },
		`"required"`: func(c *bedrock.ToolChoice) bool { return c != nil && c.Any != nil },
		`"none"`:     func(c *bedrock.ToolChoice) bool { return c == nil },
		`{"type":"function","function":{"name":"f"}}`: func(c *bedrock.ToolChoice) bool { return c != nil && c.Tool != nil && c.Tool.Name == "f" },
	}
	for raw, check := range cases {
		if got := openAIToolChoice(json.RawMessage(raw)); !check(got) {
			t.Errorf("tool_choice %s -> %#v", raw, got)
		}
	}
}

func TestConverseToOpenAISplitsTextReasoningAndTools(t *testing.T) {
	resp := &bedrock.ConverseResponse{StopReason: bedrock.StopToolUse}
	resp.Output.Message = &bedrock.Message{Role: "assistant", Content: []bedrock.ContentBlock{
		{ReasoningContent: &bedrock.ReasoningContent{ReasoningText: &bedrock.ReasoningText{Text: "thinking..."}}},
		{Text: bedrock.Ptr("here you go")},
		{ToolUse: &bedrock.ToolUseBlock{ToolUseID: "t1", Name: "f", Input: json.RawMessage(`{"a":1}`)}},
	}}
	resp.Usage = bedrock.TokenUsage{InputTokens: 5, OutputTokens: 7, TotalTokens: 12}

	out := ConverseToOpenAI(resp, "id", "model", 1)
	choice := out.Choices[0]

	if *choice.Message.Content != "here you go" {
		t.Errorf("content = %q", *choice.Message.Content)
	}
	if choice.Message.ReasoningContent != "thinking..." {
		t.Errorf("reasoning = %q", choice.Message.ReasoningContent)
	}
	if len(choice.Message.ToolCalls) != 1 || choice.Message.ToolCalls[0].Function.Arguments != `{"a":1}` {
		t.Errorf("toolCalls = %#v", choice.Message.ToolCalls)
	}
	if *choice.FinishReason != "tool_calls" {
		t.Errorf("finish = %q", *choice.FinishReason)
	}
	if out.Usage.TotalTokens != 12 {
		t.Errorf("usage = %#v", out.Usage)
	}
}

func TestOpenAIStreamSequence(t *testing.T) {
	s := NewOpenAIStream("id", "model", 0)

	var chunks []openai.ChatResponse
	for _, ev := range []bedrock.StreamEvent{
		{Type: "messageStart", Role: "assistant"},
		{Type: "contentBlockDelta", Delta: &bedrock.ContentBlockDelta{Text: "Hel"}},
		{Type: "contentBlockDelta", Delta: &bedrock.ContentBlockDelta{Text: "lo"}},
		{Type: "contentBlockStart", ContentBlockIndex: 1, Start: &bedrock.ContentBlockStart{
			ToolUse: &bedrock.ToolUseStart{ToolUseID: "t1", Name: "f"}}},
		{Type: "contentBlockDelta", ContentBlockIndex: 1, Delta: &bedrock.ContentBlockDelta{
			ToolUse: &bedrock.ToolUseDelta{Input: `{"a":`}}},
		{Type: "messageStop", StopReason: bedrock.StopToolUse},
		{Type: "metadata", Usage: &bedrock.TokenUsage{InputTokens: 3, OutputTokens: 4}},
	} {
		chunks = append(chunks, s.Handle(ev)...)
	}

	if chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Fatalf("first chunk must announce the role: %#v", chunks[0])
	}
	var text strings.Builder
	var sawTool bool
	for _, c := range chunks {
		d := c.Choices[0].Delta
		if d == nil {
			continue
		}
		if d.Content != nil {
			text.WriteString(*d.Content)
		}
		for _, tc := range d.ToolCalls {
			sawTool = true
			if tc.Index == nil || *tc.Index != 0 {
				t.Errorf("tool call index = %#v", tc.Index)
			}
		}
	}
	if text.String() != "Hello" {
		t.Errorf("streamed text = %q", text.String())
	}
	if !sawTool {
		t.Error("tool call chunks missing")
	}

	last := chunks[len(chunks)-1]
	if last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("final finish_reason = %#v", last.Choices[0].FinishReason)
	}
	if s.Usage == nil || s.Usage.OutputTokens != 4 {
		t.Errorf("usage not captured: %#v", s.Usage)
	}
}
