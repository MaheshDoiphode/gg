package logx

import (
	"fmt"
	"log"
	"os"
	"strings"
	"sync/atomic"
)

const (
	levelDebug = iota
	levelInfo
	levelWarn
	levelError
)

var (
	level int32 = levelInfo
	out         = log.New(os.Stdout, "", log.LstdFlags)
)

func SetLevel(name string) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "debug":
		atomic.StoreInt32(&level, levelDebug)
	case "warn", "warning":
		atomic.StoreInt32(&level, levelWarn)
	case "error":
		atomic.StoreInt32(&level, levelError)
	default:
		atomic.StoreInt32(&level, levelInfo)
	}
}

func enabled(l int32) bool { return atomic.LoadInt32(&level) <= l }

// DebugEnabled lets callers skip building expensive trace strings.
func DebugEnabled() bool { return enabled(levelDebug) }

func emit(tag, format string, args ...any) {
	out.Output(3, tag+" "+fmt.Sprintf(format, args...))
}

func Debugf(format string, args ...any) {
	if enabled(levelDebug) {
		emit("[DEBUG]", format, args...)
	}
}

func Infof(format string, args ...any) {
	if enabled(levelInfo) {
		emit("[INFO ]", format, args...)
	}
}

func Warnf(format string, args ...any) {
	if enabled(levelWarn) {
		emit("[WARN ]", format, args...)
	}
}

func Errorf(format string, args ...any) {
	if enabled(levelError) {
		emit("[ERROR]", format, args...)
	}
}
