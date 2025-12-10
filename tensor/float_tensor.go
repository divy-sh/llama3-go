package tensor

import (
	"encoding/binary"
	"math"
)

type MemorySegment []byte

type FloatTensor interface {
	Size() int
	GetFloat(index int) float32
	SetFloat(index int, value float32)
	Type() GGMLType

	Dot(thisOffset int, that FloatTensor, thatOffset int, size int) float32
	Matmul(context int, that []FloatTensor, out []FloatTensor, dim0 int, dim1 int)
	Reduce(thisOffset int, size int, seed float32, fn func(acc, value float32) float32) float32

	MapInPlace(thisOffset int, size int, fn func(value float32) float32) FloatTensor
	MapWithIndexInPlace(thisOffset int, size int, fn func(value float32, index int) float32) FloatTensor
	FillInPlace(thisOffset int, size int, value float32) FloatTensor

	AddInPlace(thisOffset int, that FloatTensor, thatOffset int, size int) FloatTensor
	MultiplyInPlace(thisOffset int, that FloatTensor, thatOffset int, size int) FloatTensor
	SaxpyInPlace(thisOffset int, that FloatTensor, thatOffset int, size int, a float32) FloatTensor
	CopyTo(thisOffset int, that FloatTensor, thatOffset int, size int)
}

const (
	// TODO figure out a way to enable vectorized operations conditionally
	USE_VECTOR_API = false
)

// ReadShort reads a little-endian short (int16) from the memory segment.
func ReadShort(ms MemorySegment, offset int) int16 {
	if offset+2 > len(ms) {
		panic("read short out of bounds")
	}
	return int16(binary.LittleEndian.Uint16(ms[offset : offset+2]))
}

// ReadByte reads a byte (int8) from the memory segment.
func ReadByte(ms MemorySegment, offset int) int8 {
	if offset >= len(ms) {
		panic("read byte out of bounds")
	}
	return int8(ms[offset])
}

// Float16ToFloat converts a 16-bit half-precision float to a 32-bit float.
// Note: This is a complex approximation; a proper implementation requires handling subnormals,
// infinity, and NaN according to IEEE 754-2008. The following is a fast, common approximation

func Float16ToFloat(f16 uint16) float32 {

	// We must simulate the bit manipulation for accuracy.

	s := uint32(f16>>15) & 0x1
	e := uint32(f16>>10) & 0x1f
	m := uint32(f16) & 0x3ff

	var f32Bits uint32
	if e == 0 {
		if m == 0 {
			f32Bits = s << 31 // +/- 0
		} else {
			// Subnormal: DAZ (Denormals-Are-Zero) approach
			f32Bits = s << 31 // Simplified to zero
		}
	} else if e == 0x1f {
		f32Bits = s<<31 | 0x7F800000 | (m << 13) // +/- Inf or NaN
	} else {
		// Normal: Adjust exponent bias (15) to F32 bias (127) and shift mantissa
		f32Bits = s<<31 | ((e - 15 + 127) << 23) | (m << 13)
	}

	return math.Float32frombits(f32Bits)
}

// BFloat16ToFloat converts a 16-bit bfloat16 to a 32-bit float.
func BFloat16ToFloat(bf16 int16) float32 {

	return math.Float32frombits(uint32(uint16(bf16)) << 16)
}

// ScalarDot computes the standard dot product.
func ScalarDot(thiz FloatTensor, thisOffset int, that FloatTensor, thatOffset int, size int) float32 {
	var result float32 = 0.0
	for j := 0; j < size; j++ {
		result += thiz.GetFloat(thisOffset+j) * that.GetFloat(thatOffset+j)
	}
	return result
}

// NumberOfElements calculates the total size from dimensions.
func NumberOfElements(dimensions ...int) int {
	size := 1
	for _, dim := range dimensions {
		if dim <= 0 {
			panic("dimensions must be positive")
		}
		size *= dim
	}
	return size
}

// Argmax finds the index of the maximum value.
func Argmax(t FloatTensor, thisOffset int, size int) int {
	if size <= 0 {
		return -1
	}
	maxIndex := thisOffset
	maxValue := t.GetFloat(maxIndex)
	endIndex := thisOffset + size
	for i := thisOffset + 1; i < endIndex; i++ {
		f := t.GetFloat(i)
		if f > maxValue {
			maxValue = f
			maxIndex = i
		}
	}
	return maxIndex
}

// Reduce implements the general reduction logic.
func Reduce(t FloatTensor, thisOffset int, size int, seed float32, fn func(acc, value float32) float32) float32 {
	result := seed
	for i := 0; i < size; i++ {
		result = fn(result, t.GetFloat(thisOffset+i))
	}
	return result
}

// Sum implements sum(thisOffset, size)
func Sum(t FloatTensor, thisOffset int, size int) float32 {
	return Reduce(t, thisOffset, size, 0.0, func(acc, value float32) float32 { return acc + value })
}

// Max implements max(thisOffset, size)
func Max(t FloatTensor, thisOffset int, size int) float32 {
	return Reduce(t, thisOffset, size, math.SmallestNonzeroFloat32, func(acc, value float32) float32 { return float32(math.Max(float64(acc), float64(value))) })
}

// SaxpyInPlace implements Y = a*X + Y
func SaxpyInPlace(thiz FloatTensor, thisOffset int, that FloatTensor, thatOffset int, size int, a float32) FloatTensor {
	// this[thisOffset...thisOffset+size) = a * that[thatOffset...] + this[thisOffset...]
	for i := 0; i < size; i++ {
		idx := thisOffset + i
		thatIdx := thatOffset + i
		thiz.SetFloat(idx, a*that.GetFloat(thatIdx)+thiz.GetFloat(idx))
	}
	return thiz
}

// --- Abstract Base Implementation (for common methods) ---

// BaseTensor contains common methods and delegates to the concrete implementations.
type BaseTensor struct {
	// Concrete tensor structs (Q4_0FloatTensor, ArrayFloatTensor, etc.) will embed this struct,
	// allowing them to share these methods.
	self FloatTensor // Must be set to the embedding struct instance
}

// MapInPlace implements the general mapInPlace logic.
func (b *BaseTensor) MapInPlace(thisOffset int, size int, fn func(value float32) float32) FloatTensor {
	endIndex := thisOffset + size
	for i := thisOffset; i < endIndex; i++ {
		b.self.SetFloat(i, fn(b.self.GetFloat(i)))
	}
	return b.self
}

// MapWithIndexInPlace implements the general mapWithIndexInPlace logic.
func (b *BaseTensor) MapWithIndexInPlace(thisOffset int, size int, fn func(value float32, index int) float32) FloatTensor {
	endIndex := thisOffset + size
	for i := thisOffset; i < endIndex; i++ {
		b.self.SetFloat(i, fn(b.self.GetFloat(i), i))
	}
	return b.self
}

// AddInPlace implements addition.
func (b *BaseTensor) AddInPlace(thisOffset int, that FloatTensor, thatOffset int, size int) FloatTensor {
	return b.MapWithIndexInPlace(thisOffset, size, func(value float32, index int) float32 {
		return value + that.GetFloat(index-thisOffset+thatOffset)
	})
}

// MultiplyInPlace implements element-wise multiplication.
func (b *BaseTensor) MultiplyInPlace(thisOffset int, that FloatTensor, thatOffset int, size int) FloatTensor {
	return b.MapWithIndexInPlace(thisOffset, size, func(value float32, index int) float32 {
		return value * that.GetFloat(index-thisOffset+thatOffset)
	})
}

// FillInPlace implements filling a slice with a constant value.
func (b *BaseTensor) FillInPlace(thisOffset int, size int, value float32) FloatTensor {
	return b.MapInPlace(thisOffset, size, func(unused float32) float32 { return value })
}

// CopyTo implements copying from this tensor to that tensor.
func (b *BaseTensor) CopyTo(thisOffset int, that FloatTensor, thatOffset int, size int) {
	that.MapWithIndexInPlace(thatOffset, size, func(value float32, index int) float32 {
		return b.self.GetFloat(index - thatOffset + thisOffset)
	})
}

// SoftmaxInPlace calculates the softmax of a slice.
func (b *BaseTensor) SoftmaxInPlace(thisOffset int, size int) FloatTensor {
	// Find max value (for numerical stability)
	maxVal := Max(b.self, thisOffset, size)

	// exp(x - max)
	b.MapInPlace(thisOffset, size, func(f float32) float32 { return float32(math.Exp(float64(f - maxVal))) })

	// Sum
	sumVal := Sum(b.self, thisOffset, size)

	// Normalize (divide by sum)
	return b.MapInPlace(thisOffset, size, func(f float32) float32 { return f / sumVal })
}

// Dot implements the abstract dot product using scalar computation.
func (b *BaseTensor) Dot(thisOffset int, that FloatTensor, thatOffset int, size int) float32 {
	// Quantized tensors override this to implement faster dot products when USE_VECTOR_API is true.
	return ScalarDot(b.self, thisOffset, that, thatOffset, size)
}

// Reduce implements the abstract reduce function.
func (b *BaseTensor) Reduce(thisOffset int, size int, seed float32, fn func(acc, value float32) float32) float32 {
	return Reduce(b.self, thisOffset, size, seed, fn)
}

// Matmul performs matrix multiplication (simplified for 1D tensors).
func (b *BaseTensor) Matmul(context int, that []FloatTensor, out []FloatTensor, dim0 int, dim1 int) {
	if len(that) != len(out) {
		panic("that and out arrays must have the same length")
	}

	for idxArr := 0; idxArr < len(that); idxArr++ {
		for i := 0; i < dim0; i++ {
			// out[idxArr].setFloat(i, dot(i * dim1, that[idxArr], 0, dim1))
			dotResult := b.self.Dot(i*dim1, that[idxArr], 0, dim1)
			out[idxArr].SetFloat(i, dotResult)
		}
	}
}

// SaxpyInPlace implements the Saxpy operation.
func (b *BaseTensor) SaxpyInPlace(thisOffset int, that FloatTensor, thatOffset int, size int, a float32) FloatTensor {
	return SaxpyInPlace(b.self, thisOffset, that, thatOffset, size, a)
}

// ArrayFloatTensor (F32)
type ArrayFloatTensor struct {
	BaseTensor
	values []float32
	size   int
}

func NewArrayFloatTensor(values []float32) *ArrayFloatTensor {
	t := &ArrayFloatTensor{values: values, size: len(values)}
	t.BaseTensor.self = t
	return t
}

func ArrayFloatTensorAllocate(dims ...int) *ArrayFloatTensor {
	numberOfElements := NumberOfElements(dims...)
	return NewArrayFloatTensor(make([]float32, numberOfElements))
}

func (t *ArrayFloatTensor) Size() int                         { return t.size }
func (t *ArrayFloatTensor) GetFloat(index int) float32        { return t.values[index] }
func (t *ArrayFloatTensor) SetFloat(index int, value float32) { t.values[index] = value }
func (t *ArrayFloatTensor) Type() GGMLType                    { return F32 }

// Q4_0FloatTensor
type Q4_0FloatTensor struct {
	BaseTensor
	size          int
	memorySegment MemorySegment
	TypeVal       GGMLType
}

func NewQ4_0FloatTensor(size int, memorySegment MemorySegment) *Q4_0FloatTensor {
	t := &Q4_0FloatTensor{size: size, memorySegment: memorySegment, TypeVal: Q4_0}
	t.BaseTensor.self = t
	return t
}

func (t *Q4_0FloatTensor) Size() int { return t.size }
func (t *Q4_0FloatTensor) SetFloat(index int, value float32) {
	panic("SetFloat not supported for Q4_0FloatTensor")
}
func (t *Q4_0FloatTensor) Type() GGMLType { return t.TypeVal }

func (t *Q4_0FloatTensor) GetFloat(index int) float32 {
	if index < 0 || index >= t.size {
		panic("index out of bounds")
	}

	block := Q4_0
	blockSize := block.GetBlockSize()

	blockIndex := index / blockSize
	blockOffset := blockIndex * block.ByteSizeFor(blockSize) // blockIndex * 18

	// Read scale (f16)
	scale := Float16ToFloat(binary.LittleEndian.Uint16(t.memorySegment[blockOffset : blockOffset+2]))

	var quant int8
	modIndex := index % blockSize
	quantDataOffset := blockOffset + FLOAT16_BYTES

	if modIndex < blockSize/2 { // Indices 0-15 (lower nibble of first 16 bytes)
		// Read byte, mask lower nibble (0x0F)
		byteIndex := quantDataOffset + modIndex
		byteVal := t.memorySegment[byteIndex]
		quant = int8(byteVal & 0x0F)
	} else { // Indices 16-31 (upper nibble of first 16 bytes)
		// Read byte from the offset corresponding to the first half, shift right 4, mask lower nibble (0x0F)
		byteIndex := quantDataOffset + (modIndex - blockSize/2)
		byteVal := t.memorySegment[byteIndex]
		quant = int8((byteVal >> 4) & 0x0F)
	}

	// Dequantization step: quant -= 8
	quant -= 8

	//

	return float32(quant) * scale
}

// Q8_0FloatTensor
type Q8_0FloatTensor struct {
	BaseTensor
	size          int
	memorySegment MemorySegment
	TypeVal       GGMLType
}

func NewQ8_0FloatTensor(size int, memorySegment MemorySegment) *Q8_0FloatTensor {
	t := &Q8_0FloatTensor{size: size, memorySegment: memorySegment, TypeVal: Q8_0}
	t.BaseTensor.self = t
	return t
}

func (t *Q8_0FloatTensor) Size() int { return t.size }
func (t *Q8_0FloatTensor) SetFloat(index int, value float32) {
	panic("SetFloat not supported for Q8_0FloatTensor")
}
func (t *Q8_0FloatTensor) Type() GGMLType { return t.TypeVal }

func (t *Q8_0FloatTensor) GetFloat(index int) float32 {
	if index < 0 || index >= t.size {
		panic("index out of bounds")
	}

	block := Q8_0
	blockSize := block.GetBlockSize() // 32

	blockIndex := index / blockSize
	withinBlockIndex := index % blockSize
	blockOffset := blockIndex * block.ByteSizeFor(blockSize) // blockIndex * 34

	// Read scale (f16)
	scale := Float16ToFloat(binary.LittleEndian.Uint16(t.memorySegment[blockOffset : blockOffset+2]))

	// Read quant (q8) - Data starts after scale
	quantDataOffset := blockOffset + FLOAT16_BYTES
	quant := ReadByte(t.memorySegment, quantDataOffset+withinBlockIndex)

	return float32(quant) * scale
}

// BF16FloatTensor
type BF16FloatTensor struct {
	BaseTensor
	size          int
	memorySegment MemorySegment
	TypeVal       GGMLType
}

func NewBF16FloatTensor(size int, memorySegment MemorySegment) *BF16FloatTensor {
	t := &BF16FloatTensor{size: size, memorySegment: memorySegment, TypeVal: BF16}
	t.BaseTensor.self = t
	return t
}

func (t *BF16FloatTensor) Size() int { return t.size }
func (t *BF16FloatTensor) SetFloat(index int, value float32) {
	panic("SetFloat not supported for BF16FloatTensor")
}
func (t *BF16FloatTensor) Type() GGMLType { return t.TypeVal }

func (t *BF16FloatTensor) GetFloat(index int) float32 {
	if index < 0 || index >= t.size {
		panic("index out of bounds")
	}
	offset := index * BFLOAT16_BYTES
	bf16 := ReadShort(t.memorySegment, offset)
	return BFloat16ToFloat(bf16)
}

// F16FloatTensor
type F16FloatTensor struct {
	BaseTensor
	size          int
	memorySegment MemorySegment
	TypeVal       GGMLType
}

func NewF16FloatTensor(size int, memorySegment MemorySegment) *F16FloatTensor {
	t := &F16FloatTensor{size: size, memorySegment: memorySegment, TypeVal: F16}
	t.BaseTensor.self = t
	return t
}

func (t *F16FloatTensor) Size() int { return t.size }
func (t *F16FloatTensor) SetFloat(index int, value float32) {
	panic("SetFloat not supported for F16FloatTensor")
}
func (t *F16FloatTensor) Type() GGMLType { return t.TypeVal }

func (t *F16FloatTensor) GetFloat(index int) float32 {
	if index < 0 || index >= t.size {
		panic("index out of bounds")
	}
	offset := index * FLOAT16_BYTES
	f16 := binary.LittleEndian.Uint16(t.memorySegment[offset : offset+2])
	return Float16ToFloat(f16)
}
