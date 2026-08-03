package server

import (
	"path/filepath"
	"testing"

	"bedrock-simple/internal/store"
)

// Learning a route costs a full prompt upload against the wrong endpoint, which
// was measured at 18s for a 2MB body, so it has to survive a restart.
func TestRouteCacheSurvivesRestart(t *testing.T) {
	cfg := filepath.Join(t.TempDir(), "config.json")
	if err := store.Init(cfg); err != nil {
		t.Fatal(err)
	}

	first := newRouteCache()
	first.set("xai.grok-4.3", routeResponses)

	// A second process reading the same file must start out already knowing.
	if err := store.Init(cfg); err != nil {
		t.Fatal(err)
	}
	if got := newRouteCache().get("xai.grok-4.3"); got != routeResponses {
		t.Errorf("route after restart = %q, want %q", got, routeResponses)
	}
}

func TestRouteCacheReturnsEmptyForUnknownModel(t *testing.T) {
	if err := store.Init(filepath.Join(t.TempDir(), "config.json")); err != nil {
		t.Fatal(err)
	}
	if got := newRouteCache().get("anthropic.claude-opus-5"); got != "" {
		t.Errorf("unknown model route = %q, want empty", got)
	}
}
