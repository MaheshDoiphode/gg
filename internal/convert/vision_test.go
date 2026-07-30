package convert

import (
	"strings"
	"testing"

	"bedrock-simple/internal/bedrock"
)

// The Responses API needs images as a data: URL under input_image, not as the
// raw base64 blob Converse uses.
func TestConverseToResponsesRequestImage(t *testing.T) {
	in := &bedrock.ConverseRequest{
		Messages: []bedrock.Message{{Role: "user", Content: []bedrock.ContentBlock{
			{Image: &bedrock.ImageBlock{
				Format: "png",
				Source: bedrock.ImageSource{Bytes: "iVBORw0KGgo="},
			}},
			{Text: bedrock.Ptr("what is this?")},
		}}},
	}

	out := ConverseToResponsesRequest("xai.grok-4.3", in)
	if len(out.Input) != 1 {
		t.Fatalf("input = %#v", out.Input)
	}

	var image, text *bedrock.ResponsesContent
	for i := range out.Input[0].Content {
		switch out.Input[0].Content[i].Type {
		case "input_image":
			image = &out.Input[0].Content[i]
		case "input_text":
			text = &out.Input[0].Content[i]
		}
	}
	if image == nil {
		t.Fatalf("no input_image part: %#v", out.Input[0].Content)
	}
	if want := "data:image/png;base64,iVBORw0KGgo="; image.ImageURL != want {
		t.Errorf("image_url\n got: %s\nwant: %s", image.ImageURL, want)
	}
	if text == nil || text.Text != "what is this?" {
		t.Errorf("text part = %#v", text)
	}
}

// Mantle chat completions take the same data: URL, but nested under image_url.
func TestConverseToOpenAIRequestImage(t *testing.T) {
	in := &bedrock.ConverseRequest{
		Messages: []bedrock.Message{{Role: "user", Content: []bedrock.ContentBlock{
			{Image: &bedrock.ImageBlock{
				Format: "jpeg",
				Source: bedrock.ImageSource{Bytes: "/9j/4AAQ"},
			}},
		}}},
	}

	out := ConverseToOpenAIRequest("zai.glm-5", in, false)
	raw := string(out.Messages[0].Content)
	if !strings.Contains(raw, "data:image/jpeg;base64,/9j/4AAQ") {
		t.Fatalf("content = %s", raw)
	}
	if !strings.Contains(raw, `"type":"image_url"`) {
		t.Errorf("missing image_url part: %s", raw)
	}
}

// An Anthropic image block must survive into the hub with its format intact,
// since the upstreams rebuild the data URL from it.
func TestAnthropicImageReachesHub(t *testing.T) {
	out := mustConvertAnthropic(t, `{"model":"m","max_tokens":100,"messages":[
		{"role":"user","content":[
			{"type":"image","source":{"type":"base64","media_type":"image/jpg","data":"AAAA"}},
			{"type":"text","text":"hi"}
		]}]}`)

	img := out.Messages[0].Content[0].Image
	if img == nil {
		t.Fatalf("content = %#v", out.Messages[0].Content)
	}
	// image/jpg is normalised to the format name Bedrock accepts.
	if img.Format != "jpeg" || img.Source.Bytes != "AAAA" {
		t.Errorf("image = %#v", img)
	}
}
