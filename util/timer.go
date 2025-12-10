package util

import (
	"fmt"
	"os"
	"time"
)

type Timer interface {
	Close() // No error return
}

type timerImpl struct {
	label      string
	startNanos int64
	timeUnit   time.Duration
}

// Close calculates and logs the elapsed time.
func (t *timerImpl) Close() {
	elapsedNanos := time.Now().UnixNano() - t.startNanos

	// Convert nanos to the requested unit
	var value int64
	var unitName string

	switch t.timeUnit {
	case time.Millisecond:
		value = elapsedNanos / int64(time.Millisecond)
		unitName = "milliseconds"
	case time.Microsecond:
		value = elapsedNanos / int64(time.Microsecond)
		unitName = "microseconds"
	case time.Second:
		value = elapsedNanos / int64(time.Second)
		unitName = "seconds"
	default:
		// Fallback for unexpected unit
		value = elapsedNanos
		unitName = "nanoseconds"
	}

	fmt.Fprintf(os.Stderr, "%s: %d %s\n", t.label, value, unitName)
}

// LogTimer creates and returns a Timer for logging elapsed time in milliseconds.
func LogTimer(label string) Timer {
	return LogTimerWithUnit(label, time.Millisecond)
}

// LogTimerWithUnit creates and returns a Timer for logging elapsed time in a custom unit.
func LogTimerWithUnit(label string, timeUnit time.Duration) Timer {
	return &timerImpl{
		label:      label,
		startNanos: time.Now().UnixNano(),
		timeUnit:   timeUnit,
	}
}
