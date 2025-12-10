package util

import "sync"

// ParallelFor executes a function for a range of integers concurrently.
func ParallelFor(startInclusive, endExclusive int, action func(int)) {
	if startInclusive >= endExclusive {
		return
	}
	if endExclusive-startInclusive == 1 {
		action(startInclusive)
		return
	}

	var wg sync.WaitGroup
	// Determine how many workers to use (e.g., based on GOMAXPROCS or just use goroutines freely)
	// For simplicity, we launch one goroutine per iteration, relying on Go's scheduler.

	// For large ranges, creating a goroutine for every iteration is inefficient.

	// which implicitly handles chunking/pooling.

	for i := startInclusive; i < endExclusive; i++ {
		wg.Add(1)
		i := i // Capture range variable
		go func() {
			defer wg.Done()
			action(i)
		}()
	}
	wg.Wait()
}

// ParallelForLong executes a function for a range of long integers concurrently.
func ParallelForLong(startInclusive, endExclusive int, action func(int)) {
	ParallelFor(startInclusive, endExclusive, action)
}
