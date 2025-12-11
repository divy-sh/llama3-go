package model

import (
	"math"

	"github.com/divy-sh/llama3-go/util"
)

type RoPE struct{}

// PrecomputeFreqsCis computes the RoPE frequency tables (cos/sin components) for a given context length.
// It includes the Llama 3.1 scaling logic.
func (r *RoPE) PrecomputeFreqsCis(contextLength, headSize int, theta float64,
	ropeScaling bool, scaleFactor, loFreqFactor, hiFreqFactor, oldContextLength float32) util.Pair[[]float32, []float32] {

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
			freq := float32(1.0 / math.Pow(theta, float64(i)/float64(headSize)))

			if ropeScaling {
				// Llama 3.1 scaling logic
				loFreqWavelen := oldContextLength / loFreqFactor
				hiFreqWavelen := oldContextLength / hiFreqFactor
				wavelen := float32(2.0 * math.Pi / float64(freq))

				if wavelen < hiFreqWavelen {
					// freq = freq (no change)
				} else if wavelen > loFreqWavelen {
					freq = freq / scaleFactor
				} else {
					smooth := (oldContextLength/wavelen - loFreqFactor) / (hiFreqFactor - loFreqFactor)
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
	return util.Pair[[]float32, []float32]{First: cr, Second: ci}
}
