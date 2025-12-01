package model

import (
	"math"

	"github.com/divy-sh/llama3-go/utils"
)

func PrecomputeFreqsCis(
	contextLength int,
	headSize int,
	theta float64,
	ropeScaling bool,
	scaleFactor float64,
	loFreqFactor float64,
	hiFreqFactor float64,
	oldContextLength float64) utils.Pair {
	cr := make([]float64, contextLength*(headSize/2))
	ci := make([]float64, contextLength*(headSize/2))
	n := 0
	for pos := 0; pos < contextLength; pos++ {
		for i := 0; i < headSize; i += 2 {
			freq := (float64)(1.0 / math.Pow(theta, float64(i)/float64(headSize)))
			if ropeScaling {
				// Llama 3.1 scaling
				loFreqWavelen := oldContextLength / loFreqFactor
				hiFreqWavelen := oldContextLength / hiFreqFactor
				wavelen := float64(2.0 * math.Pi / freq)
				if wavelen < hiFreqWavelen {
				} else if wavelen > loFreqWavelen {
					freq = freq / scaleFactor
				} else {
					smooth := (oldContextLength/wavelen - loFreqFactor) / (hiFreqFactor - loFreqFactor)
					freq = (1.0-smooth)*freq/scaleFactor + smooth*freq
				}
			}
			val := float64(pos) * float64(freq)
			cr[n] = float64(math.Cos(val))
			ci[n] = float64(math.Sin(val))
			n++
		}
	}
	return utils.Pair{cr, ci}
}
