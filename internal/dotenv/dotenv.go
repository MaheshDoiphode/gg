// Package dotenv loads KEY=VALUE pairs from a .env file into the process
// environment, so credentials can live in a file instead of shell exports.
package dotenv

import (
	"bufio"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// Load reads the first .env it finds next to the working directory or the
// executable and sets any variable that is not already defined. It returns the
// path it used, or "" when no file was found.
func Load() (string, error) {
	for _, path := range candidates() {
		err := loadFile(path)
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			return path, err
		}
		return path, nil
	}
	return "", nil
}

func candidates() []string {
	paths := []string{".env"}
	if exe, err := os.Executable(); err == nil {
		paths = append(paths, filepath.Join(filepath.Dir(exe), ".env"))
	}
	return paths
}

func loadFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		key, value, ok := parseLine(scanner.Text())
		if !ok {
			continue
		}
		// A real environment variable always wins over the file.
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		if err := os.Setenv(key, value); err != nil {
			return err
		}
	}
	return scanner.Err()
}

// parseLine handles `KEY=VALUE`, `export KEY=VALUE`, comments and quotes.
// The split is on the first '=' only, because base64 secrets end in '='.
func parseLine(line string) (key, value string, ok bool) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return "", "", false
	}
	line = strings.TrimPrefix(line, "export ")

	eq := strings.Index(line, "=")
	if eq <= 0 {
		return "", "", false
	}
	key = strings.TrimSpace(line[:eq])
	value = strings.TrimSpace(line[eq+1:])
	if key == "" {
		return "", "", false
	}

	if len(value) >= 2 {
		first, last := value[0], value[len(value)-1]
		if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
			return key, value[1 : len(value)-1], true
		}
	}
	// Only strip an unquoted trailing comment when it is clearly separated,
	// so values containing '#' survive.
	if i := strings.Index(value, " #"); i >= 0 {
		value = strings.TrimSpace(value[:i])
	}
	return key, value, true
}
