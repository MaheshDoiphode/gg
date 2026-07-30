package awsauth

import (
	"net/http"
	"strings"
	"testing"
	"time"
)

// TestSignVanilla checks the canonical "get-vanilla" case from the AWS
// Signature Version 4 test suite.
func TestSignVanilla(t *testing.T) {
	req, err := http.NewRequest(http.MethodGet, "https://example.amazonaws.com/", nil)
	if err != nil {
		t.Fatal(err)
	}
	when := time.Date(2015, 8, 30, 12, 36, 0, 0, time.UTC)

	Sign(req, nil, Creds{
		AccessKey: "AKIDEXAMPLE",
		SecretKey: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
	}, "us-east-1", "service", when)

	want := "AWS4-HMAC-SHA256 " +
		"Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, " +
		"SignedHeaders=host;x-amz-date, " +
		"Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"

	if got := req.Header.Get("Authorization"); got != want {
		t.Errorf("Authorization mismatch\n got: %s\nwant: %s", got, want)
	}
	if got := req.Header.Get("X-Amz-Date"); got != "20150830T123600Z" {
		t.Errorf("X-Amz-Date = %q", got)
	}
}

func TestSignIncludesSessionToken(t *testing.T) {
	req, _ := http.NewRequest(http.MethodPost, "https://bedrock-runtime.us-east-1.amazonaws.com/model/x/converse", strings.NewReader("{}"))
	req.Header.Set("Content-Type", "application/json")

	Sign(req, []byte("{}"), Creds{
		AccessKey: "AK", SecretKey: "SK", SessionToken: "TOKEN",
	}, "us-east-1", "bedrock", time.Now())

	if req.Header.Get("X-Amz-Security-Token") != "TOKEN" {
		t.Fatal("session token header missing")
	}
	auth := req.Header.Get("Authorization")
	for _, want := range []string{"content-type", "host", "x-amz-date", "x-amz-security-token"} {
		if !strings.Contains(auth, want) {
			t.Errorf("SignedHeaders missing %q: %s", want, auth)
		}
	}
}

// TestEscapePathDoubleEncoding covers the trap that breaks Bedrock signing:
// model ids contain ':', which is percent-encoded once in the request line and
// a second time in the canonical request.
func TestEscapePathDoubleEncoding(t *testing.T) {
	modelID := "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

	wire := "/model/" + EscapePathSegment(modelID) + "/converse"
	if want := "/model/us.anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse"; wire != want {
		t.Fatalf("wire path\n got: %s\nwant: %s", wire, want)
	}

	canonical := escapePath(wire, false)
	if want := "/model/us.anthropic.claude-sonnet-4-5-20250929-v1%253A0/converse"; canonical != want {
		t.Fatalf("canonical path\n got: %s\nwant: %s", canonical, want)
	}
}

func TestEscapePathLeavesUnreserved(t *testing.T) {
	if got := EscapePathSegment("aA0-_.~"); got != "aA0-_.~" {
		t.Fatalf("unreserved characters were escaped: %q", got)
	}
	if got := EscapePathSegment("a/b c"); got != "a%2Fb%20c" {
		t.Fatalf("got %q", got)
	}
}
