package tensor

import (
	"fmt"
)

type GGMLType int

const (
	F32 GGMLType = iota
	F16
	Q4_0
	Q4_1
	UNSUPPORTED_Q4_2
	UNSUPPORTED_Q4_3
	Q5_0
	Q5_1
	Q8_0
	Q8_1
	Q2_K
	Q3_K
	Q4_K
	Q5_K
	Q6_K
	Q8_K
	IQ2_XXS
	IQ2_XS
	IQ3_XXS
	IQ1_S
	IQ4_NL
	IQ3_S
	IQ2_S
	IQ4_XS
	I8
	I16
	I32
	I64
	F64
	IQ1_M
	BF16
	Q4_0_4_4
	Q4_0_4_8
	Q4_0_8_8
	TQ1_0
	TQ2_0
)

const (
	BFLOAT16_BYTES = 2
	FLOAT16_BYTES  = 2
	QK_K           = 256
)

var typeProperties = map[GGMLType]struct {
	Size  int
	Block int
}{
	F32:  {Size: 4, Block: 1},
	F16:  {Size: FLOAT16_BYTES, Block: 1},
	Q4_0: {Size: FLOAT16_BYTES + 16, Block: 32},
	Q8_0: {Size: FLOAT16_BYTES + 32, Block: 32},
	BF16: {Size: BFLOAT16_BYTES, Block: 1},
	I8:   {Size: 1, Block: 1},
	I16:  {Size: 2, Block: 1},
	I32:  {Size: 4, Block: 1},
	I64:  {Size: 8, Block: 1},
	F64:  {Size: 8, Block: 1},
}

// GetTypeSize returns the size of the block type in bytes.
func (t GGMLType) GetTypeSize() int {
	return typeProperties[t].Size
}

// GetBlockSize returns the number of elements in a block.
func (t GGMLType) GetBlockSize() int {
	return typeProperties[t].Block
}

// ByteSizeFor calculates the byte size needed for a given number of elements.
func (t GGMLType) ByteSizeFor(numberOfElements int) int {
	size := numberOfElements * t.GetTypeSize()
	blockSize := t.GetBlockSize()

	if size%blockSize != 0 {
		// This condition implies a bug in the model file or type properties.
		panic(fmt.Sprintf("internal error: size (%d) is not divisible by block size (%d) for type %v", size, blockSize, t))
	}
	return size / blockSize
}

// GGMLTensorEntry is a record holding metadata and data access for a tensor in a GGUF file.
type GGMLTensorEntry struct {
	Name     string
	GGMLType GGMLType
	Shape    []int
	Data     MemorySegment
}
