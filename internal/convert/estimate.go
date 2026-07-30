package convert

import (
	"math"
	"unicode"

	"bedrock-simple/internal/bedrock"
)

// Token counting is an estimate. Neither Mantle nor Converse exposes a counting
// endpoint for every model, and pulling in a real tokenizer would cost this
// proxy its zero-dependency property. Clients use the number for context
// budgeting, so being close is enough.

// EstimateTokens approximates the token count of a string by classifying runes:
// prose packs several characters per token, digits and symbols far fewer, and
// CJK is close to one token per character.
func EstimateTokens(s string) int {
	if s == "" {
		return 0
	}
	var ascii, digits, symbols, wide float64
	for _, r := range s {
		switch {
		case r > 0x7F:
			wide++
		case unicode.IsDigit(r):
			digits++
		case unicode.IsLetter(r) || r == ' ':
			ascii++
		default:
			symbols++
		}
	}
	total := ascii/4.5 + digits/2.0 + symbols/1.5 + wide/1.5
	if total < 1 {
		return 1
	}
	return int(math.Ceil(total))
}

// EstimateRequestTokens approximates the input size of a hub request, counting
// prompt text, tool schemas and image payloads.
func EstimateRequestTokens(req *bedrock.ConverseRequest) int {
	total := 0
	for _, s := range req.System {
		if s.Text != nil {
			total += EstimateTokens(*s.Text)
		}
	}
	for _, m := range req.Messages {
		total += 4 // per-message framing overhead
		for _, b := range m.Content {
			switch {
			case b.Text != nil:
				total += EstimateTokens(*b.Text)
			case b.ToolUse != nil:
				total += EstimateTokens(b.ToolUse.Name) + EstimateTokens(string(b.ToolUse.Input))
			case b.ToolResult != nil:
				total += EstimateTokens(toolResultText(b.ToolResult))
			case b.ReasoningContent != nil && b.ReasoningContent.ReasoningText != nil:
				total += EstimateTokens(b.ReasoningContent.ReasoningText.Text)
			case b.Image != nil:
				// Base64 is ~4 characters per 3 bytes; images bill far below
				// their encoded size, so approximate from the decoded length.
				total += len(b.Image.Source.Bytes) / 750
			}
		}
	}
	if req.ToolConfig != nil {
		for _, t := range req.ToolConfig.Tools {
			if t.ToolSpec == nil {
				continue
			}
			total += EstimateTokens(t.ToolSpec.Name) +
				EstimateTokens(t.ToolSpec.Description) +
				EstimateTokens(string(t.ToolSpec.InputSchema.JSON))
		}
	}
	return total
}
