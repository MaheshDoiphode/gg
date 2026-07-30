package dotenv

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseLine(t *testing.T) {
	cases := []struct {
		line, key, value string
		ok               bool
	}{
		{`FOO=bar`, "FOO", "bar", true},
		{`export FOO=bar`, "FOO", "bar", true},
		{`  FOO = bar  `, "FOO", "bar", true},
		{`FOO="bar baz"`, "FOO", "bar baz", true},
		{`FOO='bar baz'`, "FOO", "bar baz", true},
		{`FOO=bar # trailing`, "FOO", "bar", true},
		{`FOO=pass#word`, "FOO", "pass#word", true},
		{`# comment`, "", "", false},
		{``, "", "", false},
		{`novalue`, "", "", false},
		{`=novalue`, "", "", false},

		// A Bedrock API key is base64: it can contain '+' and '/' and end in '='.
		{`BEDROCK_API_KEY=ABSKQmVk+cm9j/aw==`, "BEDROCK_API_KEY", "ABSKQmVk+cm9j/aw==", true},
		{`AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY`,
			"AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", true},
	}

	for _, c := range cases {
		key, value, ok := parseLine(c.line)
		if ok != c.ok || key != c.key || value != c.value {
			t.Errorf("parseLine(%q) = (%q, %q, %v), want (%q, %q, %v)",
				c.line, key, value, ok, c.key, c.value, c.ok)
		}
	}
}

func TestLoadFileSkipsAlreadySetVariables(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".env")
	contents := "" +
		"# credentials\n" +
		"BEDROCK_API_KEY=ABSKfromfile==\n" +
		"AWS_REGION=eu-west-1\n" +
		"\n" +
		"export PORT=9999\n"
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("AWS_REGION", "us-east-1")
	t.Setenv("BEDROCK_API_KEY", "")
	os.Unsetenv("BEDROCK_API_KEY")
	t.Setenv("PORT", "")
	os.Unsetenv("PORT")

	if err := loadFile(path); err != nil {
		t.Fatal(err)
	}

	if got := os.Getenv("BEDROCK_API_KEY"); got != "ABSKfromfile==" {
		t.Errorf("BEDROCK_API_KEY = %q", got)
	}
	if got := os.Getenv("PORT"); got != "9999" {
		t.Errorf("PORT = %q", got)
	}
	// The real environment must win over the file.
	if got := os.Getenv("AWS_REGION"); got != "us-east-1" {
		t.Errorf("AWS_REGION = %q, the shell value should have been kept", got)
	}
}

func TestLoadMissingFileIsNotAnError(t *testing.T) {
	original, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(t.TempDir()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chdir(original) })

	path, err := Load()
	if err != nil {
		t.Fatalf("Load() = %v", err)
	}
	if path != "" {
		t.Errorf("expected no file, got %q", path)
	}
}
