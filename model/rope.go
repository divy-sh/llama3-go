package model

import (
	"math"
)

// PrecomputeFreqsCis computes the RoPE frequency tables (cos/sin components) for a given context length.
// It includes the Llama 3.1 scaling logic.
func PrecomputeFreqsCis(contextLength, headSize int, theta float32,
	ropeScaling bool, scaleFactor, loFreqFactor, hiFreqFactor float32, oldContextLength int) ([]float32, []float32) {

	if headSize%2 != 0 {
		panic("headSize must be even")
	}

	halfHeadSize := headSize / 2
	cr := make([]float32, contextLength*halfHeadSize)
	ci := make([]float32, contextLength*halfHeadSize)
	n := 0

	for pos := 0; pos < contextLength; pos++ {
		for i := 0; i < headSize; i += 2 {
			// Calculate base frequency: 1.0 / (theta ^ (i / headSize))
			freq := float32(1.0 / math.Pow(float64(theta), float64(i)/float64(headSize)))

			if ropeScaling {
				// Llama 3.1 scaling logic
				loFreqWavelen := float32(oldContextLength) / loFreqFactor
				hiFreqWavelen := float32(oldContextLength) / hiFreqFactor
				wavelen := float32(2.0 * math.Pi / float64(freq))

				if wavelen < hiFreqWavelen {
					// freq = freq (no change)
				} else if wavelen > loFreqWavelen {
					freq = freq / scaleFactor
				} else {
					smooth := (float32(oldContextLength)/wavelen - loFreqFactor) / (hiFreqFactor - loFreqFactor)
					freq = (1.0-smooth)*freq/scaleFactor + smooth*freq
				}
			}

			val := float32(pos) * freq
			cr[n] = float32(math.Cos(float64(val)))
			ci[n] = float32(math.Sin(float64(val)))
			n++
		}
	}
	if contextLength*halfHeadSize != n {
		panic("RoPE array size mismatch")
	}
	return cr, ci
}
