package infer

type Sampler interface {
	Sample(floatTensor FloatTensor) int
	Max() int
}
