package bedrock

import (
	"context"
	"errors"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"bedrock-simple/internal/logx"
	"bedrock-simple/internal/store"
)

// Registry discovers the models available to the configured credentials by
// calling the Bedrock control plane. Nothing about the catalogue is hardcoded.
type Registry struct {
	mu      sync.RWMutex
	entries []ModelEntry
	exact   map[string]string // lowercased alias -> invoke id
	fuzzy   map[string]string // punctuation-stripped alias -> invoke id
	byID    map[string]ModelEntry
	updated time.Time
	lastErr error
}

// UpstreamFor reports which API serves a resolved model id. Unknown ids default
// to Converse, which is also what raw ids and ARNs need.
func (r *Registry) UpstreamFor(id string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if e, ok := r.byID[id]; ok {
		return e.Upstream
	}
	return UpstreamConverse
}

// Upstream identifies which Bedrock API serves a model.
const (
	UpstreamConverse = "converse"
	UpstreamMantle   = "mantle"
)

// ModelEntry is one invocable model.
type ModelEntry struct {
	ID         string   // id to pass upstream: inference profile id, model id, or Mantle id
	Name       string   // human-readable name
	Provider   string   // e.g. "Anthropic"
	Region     string   // region it was discovered in
	Upstream   string   // UpstreamConverse | UpstreamMantle
	Streaming  bool     // streaming supported
	ViaProfile bool     // reached through a cross-region inference profile
	Aliases    []string // shorter names that resolve to this entry
}

// geoPrefixes are the leading segments of a cross-region inference profile id.
var geoPrefixes = map[string]bool{
	"global": true, "us": true, "eu": true, "apac": true, "apne": true,
	"ca": true, "sa": true, "jp": true, "au": true, "me": true, "af": true,
	"us-gov": true, "cn": true,
}

var (
	versionSuffix = regexp.MustCompile(`-v\d+(?::\d+)?$`)
	dateSuffix    = regexp.MustCompile(`-\d{8}$`)
	punctuation   = regexp.MustCompile(`[^a-z0-9]+`)
	datePattern   = regexp.MustCompile(`\d{8}`)
)

// ErrNoCredentialsToQuery means discovery had nothing to ask.
var ErrNoCredentialsToQuery = errors.New("no enabled credentials to discover models with")

// aliasCandidate tracks which entry currently owns a shared short alias.
type aliasCandidate struct {
	entry ModelEntry
	score int
	date  string
}

func NewRegistry() *Registry {
	return &Registry{exact: map[string]string{}, fuzzy: map[string]string{}, byID: map[string]ModelEntry{}}
}

// Resolve maps a client-supplied model name to a Bedrock invoke id.
// Order: config.json override, exact discovered alias, fuzzy alias, verbatim.
func (r *Registry) Resolve(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return ""
	}
	if target, ok := store.ModelMap()[name]; ok && target != "" {
		return target
	}

	r.mu.RLock()
	defer r.mu.RUnlock()
	if id, ok := r.exact[strings.ToLower(name)]; ok {
		return id
	}
	if id, ok := r.fuzzy[normalize(name)]; ok {
		return id
	}
	// Unknown names pass straight through, so raw ids and ARNs always work.
	return name
}

// Known reports whether an id was actually discovered, as opposed to being
// passed through unrecognised.
func (r *Registry) Known(id string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.byID[id]
	return ok
}

// ResolveWithEffort resolves a model name that may carry a -thinking-<level>
// suffix. A real model name that happens to end in -thinking is protected by
// checking the catalogue first.
func (r *Registry) ResolveWithEffort(name string) (id, effort string) {
	id = r.Resolve(name)
	if r.Known(id) {
		return id, ""
	}
	base, level, explicit := ParseEffort(name)
	if level == "" {
		return id, ""
	}
	stripped := r.Resolve(base)
	// An explicit level is always a suffix, so strip it even when discovery is
	// unavailable and nothing can be confirmed.
	if explicit || r.Known(stripped) {
		return stripped, level
	}
	return id, level
}

// Entries returns the discovered catalogue.
func (r *Registry) Entries() []ModelEntry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return append([]ModelEntry(nil), r.entries...)
}

// Status reports when the catalogue was last refreshed and the last error.
func (r *Registry) Status() (int, time.Time, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.entries), r.updated, r.lastErr
}

// Refresh rebuilds the catalogue from the control plane. It queries one
// credential per distinct region and merges the results.
func (r *Registry) Refresh(ctx context.Context, c *Client) error {
	creds := store.Credentials()
	seenRegion := map[string]bool{}

	best := map[string]*aliasCandidate{} // alias -> winner
	byID := map[string]ModelEntry{}
	var lastErr error
	queried := 0

	for _, cred := range creds {
		if !cred.Enabled || seenRegion[region(cred)] {
			continue
		}
		seenRegion[region(cred)] = true
		queried++
		geo := geoForRegion(region(cred))

		profiles, err := c.ListInferenceProfiles(ctx, cred)
		if err != nil {
			lastErr = err
			logx.Warnf("model discovery: inference profiles unavailable for %s (%s): %v", cred.Name, region(cred), err)
		}
		models, err := c.ListFoundationModels(ctx, cred)
		if err != nil {
			lastErr = err
			logx.Warnf("model discovery: foundation models unavailable for %s (%s): %v", cred.Name, region(cred), err)
		}

		// modelId -> summary, so profiles can inherit provider/streaming info.
		byModelID := make(map[string]FoundationModel, len(models))
		for _, m := range models {
			byModelID[m.ModelID] = m
		}

		add := func(e ModelEntry, score int) {
			if prev, ok := byID[e.ID]; ok {
				e.Aliases = mergeAliases(prev.Aliases, e.Aliases)
			}
			byID[e.ID] = e
			for _, a := range e.Aliases {
				key := strings.ToLower(a)
				d := dateIn(e.ID)
				cur, ok := best[key]
				if !ok || score > cur.score || (score == cur.score && d > cur.date) {
					best[key] = &aliasCandidate{entry: e, score: score, date: d}
				}
			}
		}

		for _, p := range profiles {
			if p.Status != "" && !strings.EqualFold(p.Status, "ACTIVE") {
				continue
			}
			base := baseModelID(p)
			fm, known := byModelID[base]
			if known && !isUsableTextModel(fm) {
				continue
			}
			e := ModelEntry{
				ID:         p.InferenceProfileID,
				Name:       firstNonEmpty(p.InferenceProfileName, p.InferenceProfileID),
				Provider:   providerOf(fm, p.InferenceProfileID),
				Region:     region(cred),
				Upstream:   UpstreamConverse,
				Streaming:  !known || fm.ResponseStreaming,
				ViaProfile: true,
				Aliases:    aliasesFor(p.InferenceProfileID),
			}
			add(e, profileScore(p.InferenceProfileID, geo))
		}

		for _, m := range models {
			if !isUsableTextModel(m) || !supports(m.InferenceTypesSupported, "ON_DEMAND") {
				continue
			}
			e := ModelEntry{
				ID:        m.ModelID,
				Name:      firstNonEmpty(m.ModelName, m.ModelID),
				Provider:  m.ProviderName,
				Region:    region(cred),
				Upstream:  UpstreamConverse,
				Streaming: m.ResponseStreaming,
				Aliases:   aliasesFor(m.ModelID),
			}
			add(e, 10)
		}

		// Mantle is a separate OpenAI-compatible endpoint with its own catalogue.
		mantle, err := c.ListMantleModels(ctx, cred)
		if err != nil {
			logx.Warnf("model discovery: mantle catalogue unavailable for %s (%s): %v", cred.Name, region(cred), err)
		}
		for _, m := range mantle {
			if m.ID == "" {
				continue
			}
			if m.Status != "" && !strings.EqualFold(m.Status, "available") {
				logx.Debugf("mantle model %s unavailable: %s", m.ID, m.StatusReason)
				continue
			}
			e := ModelEntry{
				ID:        m.ID,
				Name:      m.ID,
				Provider:  providerOf(FoundationModel{}, m.ID),
				Region:    region(cred),
				Upstream:  UpstreamMantle,
				Streaming: true,
				Aliases:   aliasesFor(m.ID),
			}
			add(e, mantleScore())
		}
	}

	if queried == 0 {
		return ErrNoCredentialsToQuery
	}
	if len(byID) == 0 {
		r.mu.Lock()
		r.lastErr = lastErr
		r.mu.Unlock()
		return lastErr
	}

	exact := make(map[string]string, len(best)+len(byID))
	fuzzy := make(map[string]string, len(best)+len(byID))
	for alias, cand := range best {
		exact[alias] = cand.entry.ID
		fuzzy[normalize(alias)] = cand.entry.ID
	}
	entries := make([]ModelEntry, 0, len(byID))
	for id, e := range byID {
		// A model's own id always resolves to itself, whatever won the alias.
		exact[strings.ToLower(id)] = id
		if _, taken := fuzzy[normalize(id)]; !taken {
			fuzzy[normalize(id)] = id
		}
		e.Aliases = aliasesOwnedBy(e.Aliases, id, best)
		entries = append(entries, e)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].ID < entries[j].ID })

	r.mu.Lock()
	r.entries, r.exact, r.fuzzy, r.byID = entries, exact, fuzzy, byID
	r.updated, r.lastErr = time.Now(), lastErr
	r.mu.Unlock()

	converse, mantleCount := 0, 0
	for _, e := range entries {
		if e.Upstream == UpstreamMantle {
			mantleCount++
		} else {
			converse++
		}
	}
	logx.Infof("discovered %d models across %d region(s): %d via converse, %d via mantle",
		len(entries), queried, converse, mantleCount)
	return nil
}

// StartAutoRefresh refreshes now and then every interval until ctx is done.
func (r *Registry) StartAutoRefresh(ctx context.Context, c *Client, interval time.Duration) {
	go func() {
		if err := r.Refresh(ctx, c); err != nil {
			logx.Warnf("model discovery failed: %v (unknown model names will be passed to Bedrock as-is)", err)
		}
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				if err := r.Refresh(ctx, c); err != nil {
					logx.Warnf("model discovery refresh failed: %v", err)
				}
			}
		}
	}()
}

func isUsableTextModel(m FoundationModel) bool {
	if m.ModelID == "" {
		return false
	}
	if s := m.ModelLifecycle.Status; s != "" && !strings.EqualFold(s, "ACTIVE") {
		return false
	}
	return supports(m.OutputModalities, "TEXT")
}

func supports(list []string, want string) bool {
	for _, v := range list {
		if strings.EqualFold(v, want) {
			return true
		}
	}
	return false
}

// baseModelID pulls the foundation model id out of a profile's model ARN.
func baseModelID(p InferenceProfile) string {
	for _, m := range p.Models {
		if i := strings.LastIndex(m.ModelARN, "/"); i >= 0 {
			return m.ModelARN[i+1:]
		}
	}
	return ""
}

func providerOf(m FoundationModel, id string) string {
	if m.ProviderName != "" {
		return m.ProviderName
	}
	parts := strings.Split(stripGeo(id), ".")
	if len(parts) > 1 {
		return parts[0]
	}
	return "bedrock"
}

// profileScore ranks which profile should own a shared short alias:
// global routing beats the caller's own geography, which beats everything else.
func profileScore(id, geo string) int {
	switch {
	case strings.HasPrefix(id, "global."):
		return 100
	case geo != "" && strings.HasPrefix(id, geo+"."):
		return 80
	default:
		return 50
	}
}

// mantleScore decides who wins an alias shared by both endpoints. Converse is
// richer, so it normally wins, but a Mantle-scoped API key cannot call Converse
// at all and must flip the preference. Read per refresh, since config is loaded
// after this package is initialised.
func mantleScore() int {
	if store.PreferMantle() {
		return 200
	}
	return 15
}

func geoForRegion(r string) string {
	switch {
	case strings.HasPrefix(r, "us-gov-"):
		return "us-gov"
	case strings.HasPrefix(r, "us-"):
		return "us"
	case strings.HasPrefix(r, "eu-"):
		return "eu"
	case strings.HasPrefix(r, "ap-"):
		return "apac"
	case strings.HasPrefix(r, "ca-"):
		return "ca"
	case strings.HasPrefix(r, "sa-"):
		return "sa"
	case strings.HasPrefix(r, "me-"):
		return "me"
	case strings.HasPrefix(r, "af-"):
		return "af"
	case strings.HasPrefix(r, "cn-"):
		return "cn"
	}
	return ""
}

func stripGeo(id string) string {
	if i := strings.Index(id, "."); i > 0 && geoPrefixes[id[:i]] {
		return id[i+1:]
	}
	return id
}

// aliasesFor derives progressively shorter names for a model id, e.g.
// "us.anthropic.claude-sonnet-4-5-20250929-v1:0" yields
// "anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5-20250929-v1:0",
// "claude-sonnet-4-5-20250929" and "claude-sonnet-4-5".
func aliasesFor(id string) []string {
	var out []string
	seen := map[string]bool{strings.ToLower(id): true}
	push := func(s string) {
		s = strings.TrimSpace(s)
		l := strings.ToLower(s)
		if s == "" || seen[l] {
			return
		}
		seen[l] = true
		out = append(out, s)
	}

	noGeo := stripGeo(id)
	push(noGeo)

	noProvider := noGeo
	if i := strings.Index(noGeo, "."); i > 0 {
		noProvider = noGeo[i+1:]
		push(noProvider)
	}

	noVersion := versionSuffix.ReplaceAllString(noProvider, "")
	push(noVersion)
	push(dateSuffix.ReplaceAllString(noVersion, ""))
	return out
}

// aliasesOwnedBy keeps only the aliases that actually resolve back to id.
func aliasesOwnedBy(aliases []string, id string, best map[string]*aliasCandidate) []string {
	out := make([]string, 0, len(aliases))
	for _, a := range aliases {
		if c, ok := best[strings.ToLower(a)]; ok && c.entry.ID == id {
			out = append(out, a)
		}
	}
	return out
}

func mergeAliases(a, b []string) []string {
	seen := make(map[string]bool, len(a)+len(b))
	out := make([]string, 0, len(a)+len(b))
	for _, s := range append(append([]string{}, a...), b...) {
		l := strings.ToLower(s)
		if !seen[l] {
			seen[l] = true
			out = append(out, s)
		}
	}
	return out
}

func dateIn(id string) string { return datePattern.FindString(id) }

// normalize strips punctuation so "claude-sonnet-4.5" matches "claude-sonnet-4-5".
func normalize(s string) string {
	return punctuation.ReplaceAllString(strings.ToLower(s), "")
}
