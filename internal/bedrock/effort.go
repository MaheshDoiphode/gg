package bedrock

import "strings"

// Copilot's custom-endpoint config has no reasoning control, so effort is
// selected the same way Kiro does it: with a model-name suffix. "grok-4.3",
// "grok-4.3-thinking", "grok-4.3-thinking-high" all resolve to the same model
// with different effort.
const thinkingSuffix = "-thinking"

// effortSuffixes maps the spellings clients use onto the four levels the
// endpoints accept. "max" and "xhigh" appear in Claude Code and Kiro configs.
var effortSuffixes = []struct{ suffix, level string }{
	{EffortNone, EffortNone},
	{EffortLow, EffortLow},
	{EffortMedium, EffortMedium},
	{EffortHigh, EffortHigh},
	{"max", EffortHigh},
	{"xhigh", EffortHigh},
	{"minimal", EffortNone},
}

// StripAnnotation removes a trailing bracket hint such as the [1m] context
// marker Claude Code and Kiro append. No Bedrock model id contains brackets.
func StripAnnotation(name string) string {
	trimmed := strings.TrimSpace(name)
	if !strings.HasSuffix(trimmed, "]") {
		return trimmed
	}
	if i := strings.LastIndex(trimmed, "["); i > 0 {
		return strings.TrimSpace(trimmed[:i])
	}
	return trimmed
}

// ParseEffort splits a requested model name into its base name and reasoning
// effort. An empty effort means the client did not ask for one. explicit is
// true when the level was spelled out, which no real model name does; a bare
// -thinking suffix is ambiguous with names like moonshotai.kimi-k2-thinking.
func ParseEffort(name string) (base, effort string, explicit bool) {
	trimmed := StripAnnotation(name)
	lower := strings.ToLower(trimmed)

	for _, e := range effortSuffixes {
		suffix := thinkingSuffix + "-" + e.suffix
		if strings.HasSuffix(lower, suffix) {
			return trimmed[:len(trimmed)-len(suffix)], e.level, true
		}
	}
	if strings.HasSuffix(lower, thinkingSuffix) {
		return trimmed[:len(trimmed)-len(thinkingSuffix)], EffortMedium, false
	}
	return trimmed, "", false
}

// NormalizeEffort maps arbitrary client input onto a level the endpoints accept.
// OpenAI's "minimal" and Anthropic's "disabled" both mean do not think.
func NormalizeEffort(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "none", "minimal", "off", "disabled":
		return EffortNone
	case "low":
		return EffortLow
	case "medium", "auto", "default":
		return EffortMedium
	case "high", "max", "xhigh":
		return EffortHigh
	default:
		return ""
	}
}

// EffortForBudget maps an Anthropic thinking budget onto an effort level.
func EffortForBudget(budgetTokens int) string {
	switch {
	case budgetTokens <= 0:
		return EffortNone
	case budgetTokens < 2048:
		return EffortLow
	case budgetTokens < 8192:
		return EffortMedium
	default:
		return EffortHigh
	}
}

// BudgetForEffort is the inverse, for Anthropic models on Converse which take a
// token budget rather than a level.
func BudgetForEffort(effort string) int {
	switch effort {
	case EffortLow:
		return 1024
	case EffortMedium:
		return 4096
	case EffortHigh:
		return 16384
	default:
		return 0
	}
}
