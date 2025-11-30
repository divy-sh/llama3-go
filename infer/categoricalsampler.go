package infer

import (
	"math/rand"

	"github.com/divy-sh/llama3-go/tensor/floattensor"
)

type CategoricalSampler struct {
}

func (s CategoricalSampler) Sample(logits floattensor.FloatTensor) int {
	r := rand.Float32() // random float in [0.0,1.0)
	cdf := float32(0.0)

	for i, p := range logits.Len() {
		cdf += p
		if r < cdf {
			return i
		}
	}
	return logits.Len() - 1 // fallback for rounding errors
}

func (s CategoricalSampler) Max(logits floattensor.FloatTensor) int {
	return logits.ArgMax()
}
