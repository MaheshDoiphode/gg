package bedrock

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"bedrock-simple/internal/store"
)

func TestAliasesFor(t *testing.T) {
	cases := map[string][]string{
		"us.anthropic.claude-sonnet-4-5-20250929-v1:0": {
			"anthropic.claude-sonnet-4-5-20250929-v1:0",
			"claude-sonnet-4-5-20250929-v1:0",
			"claude-sonnet-4-5-20250929",
			"claude-sonnet-4-5",
		},
		"amazon.nova-pro-v1:0": {
			"nova-pro-v1:0",
			"nova-pro",
		},
	}
	for id, want := range cases {
		got := aliasesFor(id)
		if strings.Join(got, "|") != strings.Join(want, "|") {
			t.Errorf("aliasesFor(%q)\n got: %v\nwant: %v", id, got, want)
		}
	}
}

func TestStripGeoOnlyStripsKnownPrefixes(t *testing.T) {
	if got := stripGeo("global.anthropic.claude-x"); got != "anthropic.claude-x" {
		t.Errorf("got %q", got)
	}
	// "amazon" is a provider, not a geography, and must survive.
	if got := stripGeo("amazon.nova-pro-v1:0"); got != "amazon.nova-pro-v1:0" {
		t.Errorf("got %q", got)
	}
}

func TestProfileScorePrefersGlobalThenLocalGeo(t *testing.T) {
	if profileScore("global.anthropic.x", "us") <= profileScore("us.anthropic.x", "us") {
		t.Error("global routing should outrank the local geography")
	}
	if profileScore("us.anthropic.x", "us") <= profileScore("eu.anthropic.x", "us") {
		t.Error("the caller's own geography should outrank a foreign one")
	}
}

func TestNormalizeIgnoresPunctuation(t *testing.T) {
	if normalize("claude-sonnet-4.5") != normalize("Claude_Sonnet_4_5") {
		t.Error("punctuation and case should not affect matching")
	}
}

// fakeControlPlane serves the two model-listing endpoints.
func fakeControlPlane(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasPrefix(r.URL.Path, "/inference-profiles"):
			if r.URL.Query().Get("type") != "SYSTEM_DEFINED" {
				t.Errorf("missing type filter: %s", r.URL.RawQuery)
			}
			_, _ = w.Write([]byte(`{"inferenceProfileSummaries":[
				{"inferenceProfileId":"global.anthropic.claude-sonnet-9-20991231-v1:0",
				 "inferenceProfileName":"Claude Sonnet 9","status":"ACTIVE","type":"SYSTEM_DEFINED",
				 "models":[{"modelArn":"arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-9-20991231-v1:0"}]},
				{"inferenceProfileId":"eu.anthropic.claude-sonnet-9-20991231-v1:0",
				 "inferenceProfileName":"Claude Sonnet 9 EU","status":"ACTIVE","type":"SYSTEM_DEFINED",
				 "models":[{"modelArn":"arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-sonnet-9-20991231-v1:0"}]}
			]}`))
		case strings.HasPrefix(r.URL.Path, "/foundation-models"):
			_, _ = w.Write([]byte(`{"modelSummaries":[
				{"modelId":"anthropic.claude-sonnet-9-20991231-v1:0","modelName":"Claude Sonnet 9",
				 "providerName":"Anthropic","outputModalities":["TEXT"],
				 "inferenceTypesSupported":["INFERENCE_PROFILE"],"responseStreamingSupported":true,
				 "modelLifecycle":{"status":"ACTIVE"}},
				{"modelId":"amazon.nova-pro-v1:0","modelName":"Nova Pro","providerName":"Amazon",
				 "outputModalities":["TEXT"],"inferenceTypesSupported":["ON_DEMAND"],
				 "responseStreamingSupported":true,"modelLifecycle":{"status":"ACTIVE"}},
				{"modelId":"legacy.model-v1:0","modelName":"Old","providerName":"Legacy",
				 "outputModalities":["TEXT"],"inferenceTypesSupported":["ON_DEMAND"],
				 "modelLifecycle":{"status":"LEGACY"}},
				{"modelId":"amazon.titan-image-v1:0","modelName":"Titan Image","providerName":"Amazon",
				 "outputModalities":["IMAGE"],"inferenceTypesSupported":["ON_DEMAND"],
				 "modelLifecycle":{"status":"ACTIVE"}}
			]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestRefreshDiscoversModels(t *testing.T) {
	if err := store.Init(filepath.Join(t.TempDir(), "config.json")); err != nil {
		t.Fatal(err)
	}
	srv := fakeControlPlane(t)
	if _, err := store.AddCredential(store.Credential{
		Name: "t", Enabled: true, AuthMode: store.AuthBearer,
		Region: "us-east-1", BearerKey: "bk", EndpointURL: srv.URL,
	}); err != nil {
		t.Fatal(err)
	}

	r := NewRegistry()
	if err := r.Refresh(context.Background(), New()); err != nil {
		t.Fatalf("refresh: %v", err)
	}

	// The short alias must land on the global profile, not the EU one.
	if got := r.Resolve("claude-sonnet-9"); got != "global.anthropic.claude-sonnet-9-20991231-v1:0" {
		t.Errorf("claude-sonnet-9 -> %q", got)
	}
	// Punctuation differences still resolve.
	if got := r.Resolve("Claude Sonnet 9"); got != "global.anthropic.claude-sonnet-9-20991231-v1:0" {
		t.Errorf("fuzzy match -> %q", got)
	}
	// A fully qualified id always resolves to itself.
	if got := r.Resolve("eu.anthropic.claude-sonnet-9-20991231-v1:0"); got != "eu.anthropic.claude-sonnet-9-20991231-v1:0" {
		t.Errorf("explicit id -> %q", got)
	}
	// An on-demand model with no profile is still usable.
	if got := r.Resolve("nova-pro"); got != "amazon.nova-pro-v1:0" {
		t.Errorf("nova-pro -> %q", got)
	}
	// Unknown names pass through untouched.
	if got := r.Resolve("arn:aws:bedrock:us-east-1:1:custom/x"); got != "arn:aws:bedrock:us-east-1:1:custom/x" {
		t.Errorf("passthrough -> %q", got)
	}

	for _, e := range r.Entries() {
		if strings.HasPrefix(e.ID, "legacy.") {
			t.Error("deprecated models must be filtered out")
		}
		if strings.Contains(e.ID, "titan-image") {
			t.Error("image-only models must be filtered out")
		}
	}

	count, updated, _ := r.Status()
	if count == 0 || updated.IsZero() {
		t.Errorf("status = %d %v", count, updated)
	}
}

// A real model whose name ends in -thinking must not be mistaken for an effort
// suffix, while a synthetic -thinking-<level> name must be.
func TestResolveWithEffort(t *testing.T) {
	if err := store.Init(filepath.Join(t.TempDir(), "config.json")); err != nil {
		t.Fatal(err)
	}
	srv := fakeControlPlane(t)
	if _, err := store.AddCredential(store.Credential{
		Name: "t", Enabled: true, AuthMode: store.AuthBearer,
		Region: "us-east-1", BearerKey: "bk", EndpointURL: srv.URL,
	}); err != nil {
		t.Fatal(err)
	}
	r := NewRegistry()
	if err := r.Refresh(context.Background(), New()); err != nil {
		t.Fatal(err)
	}

	// "amazon.nova-pro-thinking" is not real, so the suffix selects effort.
	id, effort := r.ResolveWithEffort("nova-pro-thinking-high")
	if id != "amazon.nova-pro-v1:0" || effort != EffortHigh {
		t.Errorf("suffix form -> (%q, %q)", id, effort)
	}

	// No suffix means no effort.
	id, effort = r.ResolveWithEffort("nova-pro")
	if id != "amazon.nova-pro-v1:0" || effort != "" {
		t.Errorf("plain form -> (%q, %q)", id, effort)
	}

	// An unknown name still passes through untouched.
	if id, _ = r.ResolveWithEffort("some.custom-model"); id != "some.custom-model" {
		t.Errorf("passthrough -> %q", id)
	}
}

// An explicit level must be stripped even with no catalogue, otherwise a
// discovery outage turns every thinking model into an invalid identifier.
func TestResolveWithEffortWithoutDiscovery(t *testing.T) {
	if err := store.Init(filepath.Join(t.TempDir(), "config.json")); err != nil {
		t.Fatal(err)
	}
	r := NewRegistry()

	id, effort := r.ResolveWithEffort("xai.grok-4.3-thinking-high")
	if id != "xai.grok-4.3" || effort != EffortHigh {
		t.Errorf("explicit level -> (%q, %q)", id, effort)
	}

	// The ambiguous bare suffix stays untouched, since it may be a real name.
	if id, _ = r.ResolveWithEffort("moonshotai.kimi-k2-thinking"); id != "moonshotai.kimi-k2-thinking" {
		t.Errorf("ambiguous suffix -> %q", id)
	}
}

func TestConfigMappingOverridesDiscovery(t *testing.T) {
	if err := store.Init(filepath.Join(t.TempDir(), "config.json")); err != nil {
		t.Fatal(err)
	}
	srv := fakeControlPlane(t)
	if _, err := store.AddCredential(store.Credential{
		Name: "t", Enabled: true, AuthMode: store.AuthBearer,
		Region: "us-east-1", BearerKey: "bk", EndpointURL: srv.URL,
	}); err != nil {
		t.Fatal(err)
	}
	r := NewRegistry()
	if err := r.Refresh(context.Background(), New()); err != nil {
		t.Fatal(err)
	}

	if err := store.SetModelMapping("claude-sonnet-9", "my.override"); err != nil {
		t.Fatal(err)
	}
	if got := r.Resolve("claude-sonnet-9"); got != "my.override" {
		t.Errorf("config override ignored: %q", got)
	}
}

func TestRefreshWithoutCredentials(t *testing.T) {
	if err := store.Init(filepath.Join(t.TempDir(), "config.json")); err != nil {
		t.Fatal(err)
	}
	r := NewRegistry()
	if err := r.Refresh(context.Background(), New()); err == nil {
		t.Fatal("expected an error when there is nothing to query")
	}
	// Resolution must still work so the proxy stays usable.
	if got := r.Resolve("anything"); got != "anything" {
		t.Errorf("passthrough broken: %q", got)
	}
}
