package bedrock

import "testing"

func TestParseEffort(t *testing.T) {
	cases := []struct {
		in, base, effort string
		explicit         bool
	}{
		{"xai.grok-4.3", "xai.grok-4.3", "", false},
		{"xai.grok-4.3-thinking", "xai.grok-4.3", EffortMedium, false},
		{"xai.grok-4.3-thinking-none", "xai.grok-4.3", EffortNone, true},
		{"xai.grok-4.3-thinking-low", "xai.grok-4.3", EffortLow, true},
		{"xai.grok-4.3-thinking-medium", "xai.grok-4.3", EffortMedium, true},
		{"xai.grok-4.3-thinking-high", "xai.grok-4.3", EffortHigh, true},
		{"Grok-Thinking-HIGH", "Grok", EffortHigh, true},
		// Claude Code and Kiro append a [1m] context hint that is not part of
		// the model id.
		{"xai.grok-4.3-thinking-high[1m]", "xai.grok-4.3", EffortHigh, true},
		{"xai.grok-4.3[1m]", "xai.grok-4.3", "", false},
		// A real name ending in -thinking is only ambiguous, never explicit, so
		// the registry can protect it.
		{"moonshotai.kimi-k2-thinking", "moonshotai.kimi-k2", EffortMedium, false},
		{"", "", "", false},
	}
	for _, c := range cases {
		base, effort, explicit := ParseEffort(c.in)
		if base != c.base || effort != c.effort || explicit != c.explicit {
			t.Errorf("ParseEffort(%q) = (%q, %q, %v), want (%q, %q, %v)",
				c.in, base, effort, explicit, c.base, c.effort, c.explicit)
		}
	}
}

func TestNormalizeEffort(t *testing.T) {
	cases := map[string]string{
		"none": EffortNone, "minimal": EffortNone, "disabled": EffortNone,
		"low": EffortLow, "LOW": EffortLow,
		"medium": EffortMedium, "auto": EffortMedium,
		"high": EffortHigh, "max": EffortHigh, "xhigh": EffortHigh,
		"nonsense": "", "": "",
	}
	for in, want := range cases {
		if got := NormalizeEffort(in); got != want {
			t.Errorf("NormalizeEffort(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestEffortBudgetRoundTrip(t *testing.T) {
	if got := EffortForBudget(0); got != EffortNone {
		t.Errorf("budget 0 -> %q", got)
	}
	if got := EffortForBudget(1024); got != EffortLow {
		t.Errorf("budget 1024 -> %q", got)
	}
	if got := EffortForBudget(4096); got != EffortMedium {
		t.Errorf("budget 4096 -> %q", got)
	}
	if got := EffortForBudget(32000); got != EffortHigh {
		t.Errorf("budget 32000 -> %q", got)
	}
	for _, level := range []string{EffortLow, EffortMedium, EffortHigh} {
		if EffortForBudget(BudgetForEffort(level)) != level {
			t.Errorf("%q did not survive the budget round trip", level)
		}
	}
	if BudgetForEffort(EffortNone) != 0 {
		t.Error("none must map to no budget")
	}
}
