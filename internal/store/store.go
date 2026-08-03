// Package store is the entire persistence layer: one JSON file on disk.
//
// It replaces the twelve DynamoDB tables used by the Python proxy. Everything
// (credentials, client API keys, model mappings, usage counters) lives in
// data/config.json, guarded by a RWMutex and written with a temp-file rename so
// a crash mid-write cannot corrupt it.
package store

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// AuthMode selects how the proxy authenticates to AWS Bedrock.
const (
	AuthBearer = "bearer" // Bedrock API key -> Authorization: Bearer <key>
	AuthSigV4  = "sigv4"  // access key / secret key -> hand-rolled SigV4
)

// Credential is one upstream Bedrock identity.
type Credential struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Enabled   bool   `json:"enabled"`
	AuthMode  string `json:"authMode"` // AuthBearer | AuthSigV4
	Region    string `json:"region"`
	BearerKey string `json:"bearerKey,omitempty"`
	AccessKey string `json:"accessKey,omitempty"`
	SecretKey string `json:"secretKey,omitempty"`
	// SessionToken is only needed for temporary STS credentials.
	SessionToken string `json:"sessionToken,omitempty"`
	// EndpointURL overrides https://bedrock-runtime.<region>.amazonaws.com.
	EndpointURL string `json:"endpointUrl,omitempty"`
	// MantleURL overrides https://bedrock-mantle.<region>.api.aws.
	MantleURL string `json:"mantleUrl,omitempty"`
	// MantleRegion points Mantle at a different region from Converse. Some
	// models are served from one region only: Grok 4.3 is us-west-2 only.
	MantleRegion string `json:"mantleRegion,omitempty"`

	CreatedAt int64 `json:"createdAt"`
	// Runtime counters.
	Requests int64  `json:"requests"`
	Failures int64  `json:"failures"`
	LastErr  string `json:"lastError,omitempty"`
	// CooldownUntil is a unix second; the picker skips this credential until then.
	CooldownUntil int64 `json:"cooldownUntil,omitempty"`
}

// APIKey is a client-facing key that callers present to this proxy.
type APIKey struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Key         string `json:"key"`
	Enabled     bool   `json:"enabled"`
	CreatedAt   int64  `json:"createdAt"`
	LastUsedAt  int64  `json:"lastUsedAt,omitempty"`
	Requests    int64  `json:"requests"`
	InputTokens int64  `json:"inputTokens"`
	OutTokens   int64  `json:"outputTokens"`
	// TokenLimit of 0 means unlimited.
	TokenLimit int64 `json:"tokenLimit"`
}

// Config is the whole on-disk document.
type Config struct {
	Host          string `json:"host"`
	Port          int    `json:"port"`
	AdminPassword string `json:"adminPassword"`
	// RequireAPIKey off makes the proxy open to anyone who can reach the port.
	RequireAPIKey bool   `json:"requireApiKey"`
	LogLevel      string `json:"logLevel"`
	DefaultRegion string `json:"defaultRegion"`
	// MaxTokensDefault caps Converse output when the client sets no limit of its
	// own. Mantle models are left uncapped so they use their full output length.
	MaxTokensDefault int `json:"maxTokensDefault"`
	// PreferMantle resolves shared model names to the Mantle endpoint instead of
	// bedrock-runtime. Needed when the key is scoped to Mantle only.
	PreferMantle bool `json:"preferMantle"`

	Credentials []Credential `json:"credentials"`
	APIKeys     []APIKey     `json:"apiKeys"`
	// ModelMap overrides/extends the built-in alias -> Bedrock model id table.
	ModelMap map[string]string `json:"modelMap"`
	// ModelRoutes remembers which Mantle API answered for a model. Learning it
	// costs a full prompt upload against the wrong endpoint, so it is persisted.
	ModelRoutes map[string]string `json:"modelRoutes"`

	TotalRequests int64 `json:"totalRequests"`
	TotalFailures int64 `json:"totalFailures"`
	TotalInTokens int64 `json:"totalInputTokens"`
	TotalOutToken int64 `json:"totalOutputTokens"`
}

var (
	mu   sync.RWMutex
	cfg  *Config
	path string
)

var (
	ErrNotFound  = errors.New("not found")
	ErrDuplicate = errors.New("duplicate")
)

func defaults() *Config {
	return &Config{
		Host:             "127.0.0.1",
		Port:             8080,
		AdminPassword:    "admin",
		RequireAPIKey:    true,
		LogLevel:         "info",
		DefaultRegion:    "us-east-1",
		MaxTokensDefault: 4096,
		Credentials:      []Credential{},
		APIKeys:          []APIKey{},
		ModelMap:         map[string]string{},
		ModelRoutes:      map[string]string{},
	}
}

// Init loads p, creating it with defaults if absent.
func Init(p string) error {
	mu.Lock()
	defer mu.Unlock()
	path = p

	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	data, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		cfg = defaults()
		return saveLocked()
	}
	if err != nil {
		return err
	}
	c := defaults()
	if err := json.Unmarshal(data, c); err != nil {
		return fmt.Errorf("parse %s: %w", p, err)
	}
	if c.Port == 0 {
		c.Port = 8080
	}
	if c.MaxTokensDefault == 0 {
		c.MaxTokensDefault = 4096
	}
	if c.DefaultRegion == "" {
		c.DefaultRegion = "us-east-1"
	}
	if c.ModelMap == nil {
		c.ModelMap = map[string]string{}
	}
	if c.ModelRoutes == nil {
		c.ModelRoutes = map[string]string{}
	}
	cfg = c
	return saveLocked()
}

// saveLocked writes via temp file + rename so a partial write never lands.
func saveLocked() error {
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// Save flushes the current in-memory config.
func Save() error {
	mu.Lock()
	defer mu.Unlock()
	return saveLocked()
}

// Path is the config file currently in use.
func Path() string {
	mu.RLock()
	defer mu.RUnlock()
	return path
}

// Snapshot returns a deep-enough copy for read-only use.
func Snapshot() Config {
	mu.RLock()
	defer mu.RUnlock()
	c := *cfg
	c.Credentials = append([]Credential(nil), cfg.Credentials...)
	c.APIKeys = append([]APIKey(nil), cfg.APIKeys...)
	c.ModelMap = make(map[string]string, len(cfg.ModelMap))
	for k, v := range cfg.ModelMap {
		c.ModelMap[k] = v
	}
	c.ModelRoutes = make(map[string]string, len(cfg.ModelRoutes))
	for k, v := range cfg.ModelRoutes {
		c.ModelRoutes[k] = v
	}
	return c
}

func Addr() string {
	mu.RLock()
	defer mu.RUnlock()
	return fmt.Sprintf("%s:%d", cfg.Host, cfg.Port)
}

// SetListen overrides the bind address without persisting it, so HOST/PORT
// env vars do not permanently rewrite the config file.
func SetListen(host string, port int) {
	mu.Lock()
	defer mu.Unlock()
	if host != "" {
		cfg.Host = host
	}
	if port > 0 {
		cfg.Port = port
	}
}

func LogLevel() string {
	mu.RLock()
	defer mu.RUnlock()
	return cfg.LogLevel
}

func AdminPassword() string {
	mu.RLock()
	defer mu.RUnlock()
	return cfg.AdminPassword
}

func SetAdminPassword(p string) error {
	mu.Lock()
	defer mu.Unlock()
	cfg.AdminPassword = p
	return saveLocked()
}

func RequireAPIKey() bool {
	mu.RLock()
	defer mu.RUnlock()
	return cfg.RequireAPIKey
}

// SetRequireAPIKey toggles client auth without persisting, so the env var does
// not permanently rewrite the config file.
func SetRequireAPIKey(v bool) {
	mu.Lock()
	defer mu.Unlock()
	cfg.RequireAPIKey = v
}

func DefaultMaxTokens() int {
	mu.RLock()
	defer mu.RUnlock()
	return cfg.MaxTokensDefault
}

func PreferMantle() bool {
	mu.RLock()
	defer mu.RUnlock()
	return cfg.PreferMantle
}

// SetPreferMantle toggles Mantle preference without persisting it.
func SetPreferMantle(v bool) {
	mu.Lock()
	defer mu.Unlock()
	cfg.PreferMantle = v
}

func ModelMap() map[string]string {
	mu.RLock()
	defer mu.RUnlock()
	out := make(map[string]string, len(cfg.ModelMap))
	for k, v := range cfg.ModelMap {
		out[k] = v
	}
	return out
}

// ModelRoutes returns the remembered Mantle route per model.
func ModelRoutes() map[string]string {
	mu.RLock()
	defer mu.RUnlock()
	out := make(map[string]string, len(cfg.ModelRoutes))
	for k, v := range cfg.ModelRoutes {
		out[k] = v
	}
	return out
}

// SetModelRoute remembers a route across restarts.
func SetModelRoute(model, route string) error {
	mu.Lock()
	defer mu.Unlock()
	if cfg.ModelRoutes == nil {
		cfg.ModelRoutes = map[string]string{}
	}
	if cfg.ModelRoutes[model] == route {
		return nil
	}
	cfg.ModelRoutes[model] = route
	return saveLocked()
}

func SetModelMapping(alias, target string) error {
	mu.Lock()
	defer mu.Unlock()
	if cfg.ModelMap == nil {
		cfg.ModelMap = map[string]string{}
	}
	if target == "" {
		delete(cfg.ModelMap, alias)
	} else {
		cfg.ModelMap[alias] = target
	}
	return saveLocked()
}

// ---------------------------------------------------------------- credentials

func Credentials() []Credential {
	mu.RLock()
	defer mu.RUnlock()
	return append([]Credential(nil), cfg.Credentials...)
}

func AddCredential(c Credential) (Credential, error) {
	mu.Lock()
	defer mu.Unlock()
	if c.ID == "" {
		c.ID = randHex(8)
	}
	for _, e := range cfg.Credentials {
		if e.ID == c.ID {
			return Credential{}, ErrDuplicate
		}
	}
	if c.Region == "" {
		c.Region = cfg.DefaultRegion
	}
	if c.AuthMode == "" {
		if c.BearerKey != "" {
			c.AuthMode = AuthBearer
		} else {
			c.AuthMode = AuthSigV4
		}
	}
	if c.Name == "" {
		c.Name = c.Region + "-" + c.ID
	}
	c.CreatedAt = time.Now().Unix()
	cfg.Credentials = append(cfg.Credentials, c)
	if err := saveLocked(); err != nil {
		cfg.Credentials = cfg.Credentials[:len(cfg.Credentials)-1]
		return Credential{}, err
	}
	return c, nil
}

func DeleteCredential(id string) error {
	mu.Lock()
	defer mu.Unlock()
	for i := range cfg.Credentials {
		if cfg.Credentials[i].ID == id {
			cfg.Credentials = append(cfg.Credentials[:i], cfg.Credentials[i+1:]...)
			return saveLocked()
		}
	}
	return ErrNotFound
}

// RecordCredentialResult updates counters and cooldown. cooldown of 0 clears it.
func RecordCredentialResult(id string, err error, cooldown time.Duration) {
	mu.Lock()
	defer mu.Unlock()
	for i := range cfg.Credentials {
		if cfg.Credentials[i].ID != id {
			continue
		}
		c := &cfg.Credentials[i]
		c.Requests++
		if err == nil {
			c.LastErr = ""
			c.CooldownUntil = 0
		} else {
			c.Failures++
			c.LastErr = truncate(err.Error(), 300)
			if cooldown > 0 {
				c.CooldownUntil = time.Now().Add(cooldown).Unix()
			}
		}
		return
	}
}

// ------------------------------------------------------------------ api keys

func APIKeys() []APIKey {
	mu.RLock()
	defer mu.RUnlock()
	return append([]APIKey(nil), cfg.APIKeys...)
}

func HasAPIKeys() bool {
	mu.RLock()
	defer mu.RUnlock()
	return len(cfg.APIKeys) > 0
}

// AddAPIKeyWithValue registers a caller-supplied key value.
func AddAPIKeyWithValue(name, value string) error {
	if value == "" {
		return errors.New("empty API key")
	}
	mu.Lock()
	defer mu.Unlock()
	for _, k := range cfg.APIKeys {
		if k.Key == value {
			return ErrDuplicate
		}
	}
	cfg.APIKeys = append(cfg.APIKeys, APIKey{
		ID: randHex(8), Name: name, Key: value, Enabled: true,
		CreatedAt: time.Now().Unix(),
	})
	if err := saveLocked(); err != nil {
		cfg.APIKeys = cfg.APIKeys[:len(cfg.APIKeys)-1]
		return err
	}
	return nil
}

func AddAPIKey(name string, tokenLimit int64) (APIKey, error) {
	mu.Lock()
	defer mu.Unlock()
	k := APIKey{
		ID:         randHex(8),
		Name:       name,
		Key:        "sk-" + randHex(32),
		Enabled:    true,
		CreatedAt:  time.Now().Unix(),
		TokenLimit: tokenLimit,
	}
	cfg.APIKeys = append(cfg.APIKeys, k)
	if err := saveLocked(); err != nil {
		cfg.APIKeys = cfg.APIKeys[:len(cfg.APIKeys)-1]
		return APIKey{}, err
	}
	return k, nil
}

// FindAPIKey looks a presented key up by value. Comparison is constant-time.
func FindAPIKey(value string) *APIKey {
	mu.RLock()
	defer mu.RUnlock()
	for i := range cfg.APIKeys {
		if constantTimeEqual(cfg.APIKeys[i].Key, value) {
			k := cfg.APIKeys[i]
			return &k
		}
	}
	return nil
}

func OverLimit(k APIKey) bool {
	return k.TokenLimit > 0 && k.InputTokens+k.OutTokens >= k.TokenLimit
}

// RecordUsage attributes tokens to a key and to the global counters.
func RecordUsage(keyID string, in, out int64, failed bool) {
	mu.Lock()
	defer mu.Unlock()
	cfg.TotalRequests++
	if failed {
		cfg.TotalFailures++
	}
	cfg.TotalInTokens += in
	cfg.TotalOutToken += out
	for i := range cfg.APIKeys {
		if cfg.APIKeys[i].ID == keyID {
			cfg.APIKeys[i].Requests++
			cfg.APIKeys[i].InputTokens += in
			cfg.APIKeys[i].OutTokens += out
			cfg.APIKeys[i].LastUsedAt = time.Now().Unix()
			break
		}
	}
}

func randHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		// crypto/rand failing is unrecoverable for a key-issuing service.
		panic("crypto/rand unavailable: " + err.Error())
	}
	return hex.EncodeToString(b)
}

func constantTimeEqual(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	var v byte
	for i := 0; i < len(a); i++ {
		v |= a[i] ^ b[i]
	}
	return v == 0
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return strings.TrimSpace(s[:n]) + "..."
}
