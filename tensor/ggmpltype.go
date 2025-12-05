package tensor

import "math"

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
	QK_K           = 256 // or 64?
)

// TypeInfo stores per-enum metadata like Java's fields.
type TypeInfo struct {
	TypeSize  int
	BlockSize int
}

var ggmlTypes = map[GGMLType]TypeInfo{
	F32:              {4, 1},
	F16:              {FLOAT16_BYTES, 1},
	Q4_0:             {FLOAT16_BYTES + 16, 32},
	Q4_1:             {2*FLOAT16_BYTES + 16, 32},
	UNSUPPORTED_Q4_2: {math.Maxint, 1},
	UNSUPPORTED_Q4_3: {math.Maxint, 1},
	Q5_0:             {math.Maxint, 1},
	Q5_1:             {math.Maxint, 1},
	Q8_0:             {FLOAT16_BYTES + 32, 32},
	Q8_1:             {32 + 2*4, 32},
	Q2_K:             {math.Maxint, 1},
	Q3_K:             {math.Maxint, 1},
	Q4_K:             {2*FLOAT16_BYTES + ((QK_K / 16) / 8 * 6) + QK_K/2, QK_K},
	Q5_K:             {2*FLOAT16_BYTES + ((QK_K / 16) / 8 * 6) + QK_K/8 + QK_K/2, QK_K},
	Q6_K:             {QK_K/2 + QK_K/4 + QK_K/16 + FLOAT16_BYTES, QK_K},
	Q8_K:             {math.Maxint, 1},
	IQ2_XXS:          {math.Maxint, 1},
	IQ2_XS:           {math.Maxint, 1},
	IQ3_XXS:          {math.Maxint, 1},
	IQ1_S:            {math.Maxint, 1},
	IQ4_NL:           {math.Maxint, 1},
	IQ3_S:            {math.Maxint, 1},
	IQ2_S:            {math.Maxint, 1},
	IQ4_XS:           {math.Maxint, 1},
	I8:               {1, 1},
	I16:              {2, 1},
	I32:              {4, 1},
	I64:              {8, 1},
	F64:              {8, 1},
	IQ1_M:            {math.Maxint, 1},
	BF16:             {BFLOAT16_BYTES, 1},
	Q4_0_4_4:         {FLOAT16_BYTES + 16, 32},
	Q4_0_4_8:         {FLOAT16_BYTES + 16, 32},
	Q4_0_8_8:         {FLOAT16_BYTES + 16, 32},
	TQ1_0:            {math.Maxint, 1},
	TQ2_0:            {math.Maxint, 1},
}

// Functions matching Java methods
func (t GGMLType) TypeSize() int {
	return ggmlTypes[t].TypeSize
}

func (t GGMLType) BlockSize() int {
	return ggmlTypes[t].BlockSize
}

func FromID(id int) GGMLType {
	return GGMLType(id)
}

// byteSizeFor = (elements * typeSize) / blockSize
func (t GGMLType) ByteSizeFor(n int) int {
	ti := ggmlTypes[t]
	total := n * ti.TypeSize

	if ti.BlockSize <= 0 || (ti.BlockSize&(ti.BlockSize-1)) != 0 {
		panic("block size must be a positive power of 2")
	}
	if total%ti.BlockSize != 0 {
		panic("total size not divisible by block size")
	}

	return total / ti.BlockSize
}
