// Command bedrock-simple is a dependency-free proxy that exposes AWS Bedrock
// through OpenAI- and Anthropic-compatible HTTP APIs.
package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"bedrock-simple/internal/bedrock"
	"bedrock-simple/internal/dotenv"
	"bedrock-simple/internal/logx"
	"bedrock-simple/internal/server"
	"bedrock-simple/internal/store"
)

const version = "1.0.0"

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "fatal:", err)
		os.Exit(1)
	}
}

func run() error {
	// Loaded first so everything below, including CONFIG_PATH, can come from it.
	envFile, err := dotenv.Load()
	if err != nil {
		return fmt.Errorf("load .env: %w", err)
	}

	configPath := envOr("CONFIG_PATH", "data/config.json")
	if err := store.Init(configPath); err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	applyEnvOverrides()
	logx.SetLevel(envOr("LOG_LEVEL", store.LogLevel()))

	if err := adoptEnvCredential(); err != nil {
		return err
	}
	clientKey, err := ensureClientKey()
	if err != nil {
		return err
	}

	client := bedrock.New()
	registry := bedrock.NewRegistry()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	registry.StartAutoRefresh(ctx, client, 6*time.Hour)

	addr := store.Addr()
	srv := &http.Server{
		Addr:              addr,
		Handler:           server.New(client, registry),
		ReadHeaderTimeout: 30 * time.Second,
		ReadTimeout:       10 * time.Minute,
		IdleTimeout:       120 * time.Second,
		// WriteTimeout stays zero so long SSE streams are never cut off.
	}

	printBanner(addr, configPath, clientKey, envFile)

	errCh := make(chan error, 1)
	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
	return store.Save()
}

func printBanner(addr, configPath, clientKey, envFile string) {
	creds := store.Credentials()
	base := "http://" + addr
	if strings.HasPrefix(addr, "0.0.0.0:") {
		base = "http://localhost:" + strings.TrimPrefix(addr, "0.0.0.0:")
	}

	fmt.Printf("\nbedrock-simple %s  ->  %s\n\n", version, base)
	fmt.Println("  OpenAI compatible      POST  /v1/chat/completions")
	fmt.Println("                         GET   /v1/models")
	fmt.Println("  Anthropic compatible   POST  /v1/messages")
	fmt.Println("  Health                 GET   /health")
	fmt.Println()

	if len(creds) == 0 {
		fmt.Println("  credentials  NONE - set BEDROCK_API_KEY, or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY,")
		fmt.Println("               in a .env file next to this binary, or add one under \"credentials\" in")
		fmt.Printf("               %s\n", configPath)
	} else {
		for _, c := range creds {
			state := "enabled"
			if !c.Enabled {
				state = "disabled"
			}
			fmt.Printf("  credential   %-16s %-6s %-12s %s\n", c.Name, c.AuthMode, c.Region, state)
		}
	}
	if envFile != "" {
		fmt.Printf("  env file     %s\n", envFile)
	}

	switch {
	case !store.RequireAPIKey():
		fmt.Println("  client auth  DISABLED - anyone who can reach this port can use it")
	case clientKey != "":
		fmt.Printf("  client key   %s\n", clientKey)
	default:
		fmt.Printf("  client auth  required (%d key(s) in %s)\n", len(store.APIKeys()), configPath)
	}

	fmt.Println("\n  Only failures are logged below.")
	fmt.Println()
}

// applyEnvOverrides lets HOST/PORT/LOG_LEVEL win over the config file.
func applyEnvOverrides() {
	cfg := store.Snapshot()
	if v := os.Getenv("PORT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Port = n
		}
	}
	if v := os.Getenv("HOST"); v != "" {
		cfg.Host = v
	}
	if v := os.Getenv("REQUIRE_API_KEY"); v != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "0", "false", "no", "off":
			store.SetRequireAPIKey(false)
		case "1", "true", "yes", "on":
			store.SetRequireAPIKey(true)
		}
	}
	if v := os.Getenv("PREFER_MANTLE"); v != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "on":
			store.SetPreferMantle(true)
		case "0", "false", "no", "off":
			store.SetPreferMantle(false)
		}
	}
	if cfg.Port != 0 {
		store.SetListen(cfg.Host, cfg.Port)
	}
}

// adoptEnvCredential registers a credential from the environment on first run,
// so the proxy works with no config file and no `aws configure`.
func adoptEnvCredential() error {
	if len(store.Credentials()) > 0 {
		return nil
	}
	region := firstEnv("AWS_REGION", "AWS_DEFAULT_REGION", "BEDROCK_REGION")
	if region == "" {
		region = "us-east-1"
	}

	if bearer := firstEnv("BEDROCK_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"); bearer != "" {
		_, err := store.AddCredential(store.Credential{
			Name: "env-bearer", Enabled: true, AuthMode: store.AuthBearer,
			Region: region, BearerKey: bearer,
			MantleRegion: firstEnv("MANTLE_REGION"),
		})
		return err
	}
	ak, sk := os.Getenv("AWS_ACCESS_KEY_ID"), os.Getenv("AWS_SECRET_ACCESS_KEY")
	if ak != "" && sk != "" {
		_, err := store.AddCredential(store.Credential{
			Name: "env-sigv4", Enabled: true, AuthMode: store.AuthSigV4,
			Region: region, AccessKey: ak, SecretKey: sk,
			SessionToken: os.Getenv("AWS_SESSION_TOKEN"),
			MantleRegion: firstEnv("MANTLE_REGION"),
		})
		return err
	}
	return nil
}

// ensureClientKey returns a key to print in the banner, minting one on first
// run so the proxy is never left both required-auth and unusable.
func ensureClientKey() (string, error) {
	if !store.RequireAPIKey() {
		return "", nil
	}
	// PROXY_API_KEY is authoritative on every run, not just the first, so the
	// key you configure in a client stays valid after data/config.json exists.
	if v := os.Getenv("PROXY_API_KEY"); v != "" {
		if store.FindAPIKey(v) == nil {
			if err := store.AddAPIKeyWithValue("env", v); err != nil {
				return "", err
			}
		}
		return v, nil
	}
	if store.HasAPIKeys() {
		return "", nil
	}
	k, err := store.AddAPIKey("default", 0)
	if err != nil {
		return "", err
	}
	return k.Key, nil
}

func envOr(name, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return fallback
}

func firstEnv(names ...string) string {
	for _, n := range names {
		if v := strings.TrimSpace(os.Getenv(n)); v != "" {
			return v
		}
	}
	return ""
}
