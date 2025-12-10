package model

import (
	"math/rand"
	"sort"

	"github.com/divy-sh/llama3-go/tensor"
)

type Sampler interface {
	SampleToken(logits tensor.FloatTensor) int
}

type ArgmaxSampler struct{}

var ARGMAX Sampler = ArgmaxSampler{}

func (s ArgmaxSampler) SampleToken(logits tensor.FloatTensor) int {
	return tensor.Argmax(logits, 0, logits.Size())
}

type CategoricalSampler struct {
	Rng *rand.Rand
}

func (s *CategoricalSampler) SampleToken(logits tensor.FloatTensor) int {
	// Sample a float from [0.0, 1.0)
	random0to1 := s.Rng.Float32()
	var cdf float32 = 0.0

	for i := 0; i < logits.Size(); i++ {
		cdf += logits.GetFloat(i)
		if random0to1 < cdf {
			return i
		}
	}
	// In case of rounding errors, return the last token
	return logits.Size() - 1
}

// ToppSampler implements Top-P (Nucleus) sampling.
type ToppSampler struct {
	indices []int
	topp    float32
	rng     *rand.Rand
}

func NewToppSampler(maxNumberOfElements int, topp float32, rng *rand.Rand) *ToppSampler {
	// We need a maximum of maxNumberOfElements indices to store and sort.
	return &ToppSampler{
		indices: make([]int, maxNumberOfElements),
		topp:    topp,
		rng:     rng,
	}
}

type tokenHeap struct {
	indices []int
	logits  tensor.FloatTensor
	size    int
}

func (h *tokenHeap) Less(i, j int) bool {
	return h.logits.GetFloat(h.indices[i]) < h.logits.GetFloat(h.indices[j])
}

func (h *tokenHeap) Swap(i, j int) {
	h.indices[i], h.indices[j] = h.indices[j], h.indices[i]
}

func (h *tokenHeap) Pop() int {
	h.size--
	return h.indices[h.size]
}

func (h *tokenHeap) Push(x int) {
	h.indices[h.size] = x
	h.size++
}

func MaxHeapSort(indices []int, logits tensor.FloatTensor, n0 int) {
	sort.Slice(indices[:n0], func(i, j int) bool {
		return logits.GetFloat(indices[i]) > logits.GetFloat(indices[j])
	})
}

func (s *ToppSampler) SampleToken(logits tensor.FloatTensor) int {
	n := logits.Size()

	// Pre-filter by cutoff
	head := 0
	// values smaller than (1 - topp) / (n - 1) cannot be part of the result
	cutoff := (1.0 - s.topp) / float32(n-1)

	maxIndices := len(s.indices)
	if n < maxIndices {
		maxIndices = n
	}

	for i := 0; i < maxIndices; i++ {
		if logits.GetFloat(i) >= cutoff {
			s.indices[head] = i
			head++
		} else {
			// For now, we only care about `head` and rely on a full sort
			// instead of the partial sort/siftDown combo.
			// TODO fix this later for performance.
		}
	}

	n0 := head // Number of candidates after cutoff

	MaxHeapSort(s.indices, logits, n0)

	var cumulativeProb float32 = 0.0
	lastIndex := n0 - 1 // Default to the smallest index in the sorted list (last element)

	for i := 0; i < n0; i++ {
		cumulativeProb += logits.GetFloat(s.indices[i])
		if cumulativeProb > s.topp {
			// This index `i` is the last one to include
			lastIndex = i
			break
		}
	}

	// If cumulativeProb <= topp even after including all candidates, we take all n0 tokens.
	if cumulativeProb <= s.topp {
		lastIndex = n0 - 1
	}

	// `cumulativeProb` now holds the sum of probabilities of the selected tokens [0...lastIndex]
	r := s.rng.Float32() * cumulativeProb
	var cdf float32 = 0.0

	// Sample from indices[0] up to indices[lastIndex]
	for i := 0; i <= lastIndex; i++ {
		cdf += logits.GetFloat(s.indices[i])
		if r < cdf {
			return s.indices[i]
		}
	}

	// In case of rounding errors, return the last token in the truncated list (smallest prob)
	return s.indices[lastIndex]
}
