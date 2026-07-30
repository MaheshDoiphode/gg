// Package awsauth implements AWS Signature Version 4 with nothing but the
// standard library, so the proxy needs no AWS SDK and no `aws configure`.
package awsauth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

// Creds are the static credentials used for signing. SessionToken is optional
// and only present for temporary (STS) credentials.
type Creds struct {
	AccessKey    string
	SecretKey    string
	SessionToken string
}

const (
	algorithm    = "AWS4-HMAC-SHA256"
	timeFormat   = "20060102T150405Z"
	dateFormat   = "20060102"
	unsignedBody = "UNSIGNED-PAYLOAD"
)

// Sign adds the SigV4 Authorization header to req in place. payload must be the
// exact bytes of the request body (may be nil for bodyless requests).
func Sign(req *http.Request, payload []byte, creds Creds, region, service string, now time.Time) {
	now = now.UTC()
	amzDate := now.Format(timeFormat)
	scopeDate := now.Format(dateFormat)

	payloadHash := hexSHA256(payload)

	req.Header.Set("X-Amz-Date", amzDate)
	if creds.SessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", creds.SessionToken)
	}
	if req.Host != "" {
		req.Header.Set("Host", req.Host)
	} else {
		req.Header.Set("Host", req.URL.Host)
	}

	signedHeaders, canonicalHeaders := canonicalizeHeaders(req)

	// AWS canonicalises the *already URI-encoded* path a second time for every
	// service except S3. Bedrock model ids contain ':' so this matters.
	canonicalURI := escapePath(rawPath(req), false)
	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI,
		canonicalQuery(req),
		canonicalHeaders,
		signedHeaders,
		payloadHash,
	}, "\n")

	scope := strings.Join([]string{scopeDate, region, service, "aws4_request"}, "/")
	stringToSign := strings.Join([]string{
		algorithm,
		amzDate,
		scope,
		hexSHA256([]byte(canonicalRequest)),
	}, "\n")

	signingKey := deriveKey(creds.SecretKey, scopeDate, region, service)
	signature := hex.EncodeToString(hmacSHA256(signingKey, []byte(stringToSign)))

	req.Header.Set("Authorization", fmt.Sprintf(
		"%s Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		algorithm, creds.AccessKey, scope, signedHeaders, signature))
}

func rawPath(req *http.Request) string {
	p := req.URL.EscapedPath()
	if p == "" {
		return "/"
	}
	return p
}

// canonicalizeHeaders returns the signed-header list and the canonical header
// block. Only a fixed safe set is signed, which keeps the signature stable even
// if Go's transport adds headers later.
func canonicalizeHeaders(req *http.Request) (string, string) {
	names := []string{"host"}
	values := map[string]string{"host": req.Header.Get("Host")}

	for name := range req.Header {
		l := strings.ToLower(name)
		if l == "host" || l == "authorization" || l == "content-length" {
			continue
		}
		if !strings.HasPrefix(l, "x-amz") && l != "content-type" {
			continue
		}
		names = append(names, l)
		values[l] = strings.TrimSpace(req.Header.Get(name))
	}
	sort.Strings(names)

	var b strings.Builder
	for _, n := range names {
		b.WriteString(n)
		b.WriteByte(':')
		b.WriteString(collapseSpaces(values[n]))
		b.WriteByte('\n')
	}
	return strings.Join(names, ";"), b.String()
}

func canonicalQuery(req *http.Request) string {
	q := req.URL.Query()
	keys := make([]string, 0, len(q))
	for k := range q {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		vals := append([]string(nil), q[k]...)
		sort.Strings(vals)
		for _, v := range vals {
			parts = append(parts, escapeQuery(k)+"="+escapeQuery(v))
		}
	}
	return strings.Join(parts, "&")
}

func deriveKey(secret, date, region, service string) []byte {
	k := hmacSHA256([]byte("AWS4"+secret), []byte(date))
	k = hmacSHA256(k, []byte(region))
	k = hmacSHA256(k, []byte(service))
	return hmacSHA256(k, []byte("aws4_request"))
}

func hmacSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}

func hexSHA256(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func isUnreserved(c byte) bool {
	return c >= 'A' && c <= 'Z' ||
		c >= 'a' && c <= 'z' ||
		c >= '0' && c <= '9' ||
		c == '-' || c == '_' || c == '.' || c == '~'
}

// EscapePathSegment percent-encodes a single path segment (encoding '/' too).
func EscapePathSegment(s string) string { return escapePath(s, true) }

func escapePath(s string, encodeSlash bool) string {
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if isUnreserved(c) || (c == '/' && !encodeSlash) {
			b.WriteByte(c)
			continue
		}
		fmt.Fprintf(&b, "%%%02X", c)
	}
	return b.String()
}

func escapeQuery(s string) string { return escapePath(s, true) }

func collapseSpaces(s string) string {
	if !strings.Contains(s, "  ") {
		return s
	}
	return strings.Join(strings.Fields(s), " ")
}
