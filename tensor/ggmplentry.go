package tensor

type GGMLTensorEntry struct {
	MappedFile    []byte
	Name          string
	Type          GGMLType
	Shape         []int
	MemorySegment []byte
}
