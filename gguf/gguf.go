package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/divy-sh/llama3-go/tensor"
	"github.com/divy-sh/llama3-go/util"
)

const (
	GGUF_MAGIC        = 0x46554747
	DEFAULT_ALIGNMENT = 32 // must be a power of 2
)

var SUPPORTED_GGUF_VERSIONS = []int{2, 3}

// GGUF struct holds the parsed GGUF file structure.
type GGUF struct {
	Magic            uint32
	Version          uint32
	TensorCount      uint64
	MetadataKVCount  uint64
	Metadata         map[string]interface{}
	TensorInfos      map[string]GGUFTensorInfo
	Alignment        int
	TensorDataOffset int64
}

// GGUFTensorInfo maps to gguf_tensor_info_t.
type GGUFTensorInfo struct {
	Name       string
	Dimensions []int
	GGMLType   tensor.GGMLType
	Offset     uint64
}

// MetadataValueType is an enum for GGUF metadata value types.
type MetadataValueType int

const (
	MetadataValueTypeUINT8 MetadataValueType = iota
	MetadataValueTypeINT8
	MetadataValueTypeUINT16
	MetadataValueTypeINT16
	MetadataValueTypeUINT32
	MetadataValueTypeINT32
	MetadataValueTypeFLOAT32
	MetadataValueTypeBOOL
	MetadataValueTypeSTRING
	MetadataValueTypeARRAY
	MetadataValueTypeUINT64
	MetadataValueTypeINT64
	MetadataValueTypeFLOAT64
)

// Size returns the byte size of the primitive type. Negative for variable-length types.
func (t MetadataValueType) Size() int {
	switch t {
	case MetadataValueTypeUINT8, MetadataValueTypeINT8, MetadataValueTypeBOOL:
		return 1
	case MetadataValueTypeUINT16, MetadataValueTypeINT16:
		return 2
	case MetadataValueTypeUINT32, MetadataValueTypeINT32, MetadataValueTypeFLOAT32:
		return 4
	case MetadataValueTypeUINT64, MetadataValueTypeINT64, MetadataValueTypeFLOAT64:
		return 8
	case MetadataValueTypeSTRING, MetadataValueTypeARRAY:
		return -8 // Represents the size of the prepended length field (uint64)
	default:
		panic(fmt.Sprintf("Unknown MetadataValueType: %d", t))
	}
}

// ByteOrder for GGUF is always LittleEndian.
var byteOrder = binary.LittleEndian

// LoadModel opens the file and parses the GGUF structure.
func LoadModel(modelPath string) (*GGUF, error) {
	file, err := os.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()
	defer util.LogTimer(fmt.Sprintf("Parse %s", modelPath)).Close()

	gguf := &GGUF{}
	if err := gguf.loadModelImpl(file); err != nil {
		return nil, err
	}
	return gguf, nil
}

func (g *GGUF) loadModelImpl(r io.ReadSeeker) error {
	// Read Header and Metadata
	if err := g.readHeader(r); err != nil {
		return err
	}

	// Read Tensor Infos
	g.TensorInfos = make(map[string]GGUFTensorInfo, int(g.TensorCount))
	for i := 0; i < int(g.TensorCount); i++ {
		ti, err := g.readTensorInfo(r)
		if err != nil {
			return fmt.Errorf("failed to read tensor info %d: %w", i, err)
		}
		if _, exists := g.TensorInfos[ti.Name]; exists {
			return fmt.Errorf("duplicate tensor name: %s", ti.Name)
		}
		g.TensorInfos[ti.Name] = ti
	}

	// Padding to nearest multiple of ALIGNMENT
	currentPos, _ := r.Seek(0, io.SeekCurrent)
	padding := int64(g.GetAlignment()) - (currentPos % int64(g.GetAlignment()))
	if padding != int64(g.GetAlignment()) { // Only seek if actual padding is needed
		currentPos += padding
		if _, err := r.Seek(currentPos, io.SeekStart); err != nil {
			return fmt.Errorf("failed to seek past padding: %w", err)
		}
	}

	// Tensor Data Offset
	g.TensorDataOffset = currentPos

	return nil
}

// LoadTensors opens the GGUF file and loads tensor data based on tensor infos
func LoadTensors(modelPath string, tensorDataOffset int64, tensorInfos map[string]GGUFTensorInfo) (map[string]tensor.GGMLTensorEntry, error) {
	file, err := os.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file: %w", err)
	}
	defer file.Close()

	result := make(map[string]tensor.GGMLTensorEntry, len(tensorInfos))

	for name, info := range tensorInfos {
		// Calculate the number of elements
		numElements := 1
		for _, dim := range info.Dimensions {
			numElements *= dim
		}

		// Calculate the byte size needed for this tensor
		byteSize := info.GGMLType.ByteSizeFor(numElements)

		// Seek to the tensor data location (tensorDataOffset + offset)
		tensorOffset := tensorDataOffset + int64(info.Offset)
		if _, err := file.Seek(tensorOffset, io.SeekStart); err != nil {
			return nil, fmt.Errorf("failed to seek to tensor '%s' data: %w", name, err)
		}

		// Read the tensor data
		data := make([]byte, byteSize)
		if _, err := io.ReadFull(file, data); err != nil {
			return nil, fmt.Errorf("failed to read tensor '%s' data: %w", name, err)
		}

		// Create GGMLTensorEntry
		result[name] = tensor.GGMLTensorEntry{
			Name:     name,
			GGMLType: info.GGMLType,
			Shape:    info.Dimensions,
			Data:     tensor.MemorySegment(data),
		}
	}

	return result, nil
}

func (g *GGUF) readHeader(r io.Reader) error {
	// Magic
	if err := binary.Read(r, byteOrder, &g.Magic); err != nil {
		return fmt.Errorf("failed to read magic: %w", err)
	}
	if g.Magic != GGUF_MAGIC {
		return fmt.Errorf("unsupported header.magic 0x%X", g.Magic)
	}

	// Version
	if err := binary.Read(r, byteOrder, &g.Version); err != nil {
		return fmt.Errorf("failed to read version: %w", err)
	}
	found := false
	for _, v := range SUPPORTED_GGUF_VERSIONS {
		if int(g.Version) == v {
			found = true
			break
		}
	}
	if !found {
		return fmt.Errorf("unsupported header.version %d", g.Version)
	}

	// TensorCount
	if err := binary.Read(r, byteOrder, &g.TensorCount); err != nil {
		return fmt.Errorf("failed to read tensor_count: %w", err)
	}

	// MetadataKVCount
	if err := binary.Read(r, byteOrder, &g.MetadataKVCount); err != nil {
		return fmt.Errorf("failed to read metadata_kv_count: %w", err)
	}

	// Metadata Key-Value pairs
	g.Metadata = make(map[string]interface{}, int(g.MetadataKVCount))
	for i := 0; i < int(g.MetadataKVCount); i++ {
		key, value, err := g.readKeyValuePair(r)
		if err != nil {
			return fmt.Errorf("failed to read metadata key-value pair %d: %w", i, err)
		}
		if _, exists := g.Metadata[key]; exists {
			return fmt.Errorf("duplicate metadata key: %s", key)
		}
		g.Metadata[key] = value
	}

	return nil
}

func (g *GGUF) readTensorInfo(r io.Reader) (GGUFTensorInfo, error) {
	// Name
	name, err := g.readString(r)
	if err != nil {
		return GGUFTensorInfo{}, fmt.Errorf("failed to read tensor name: %w", err)
	}
	if len(name) > 64 {
		return GGUFTensorInfo{}, fmt.Errorf("tensor name length > 64: %s", name)
	}

	// Number of dimensions
	var nDimensions uint32
	if err := binary.Read(r, byteOrder, &nDimensions); err != nil {
		return GGUFTensorInfo{}, fmt.Errorf("failed to read n_dimensions: %w", err)
	}
	if nDimensions > 4 {
		return GGUFTensorInfo{}, fmt.Errorf("unsupported number of dimensions: %d", nDimensions)
	}

	// Dimensions
	dimensions := make([]int, nDimensions)
	for i := 0; i < int(nDimensions); i++ {
		var dim uint64
		if err := binary.Read(r, byteOrder, &dim); err != nil {
			return GGUFTensorInfo{}, fmt.Errorf("failed to read dimension %d: %w", i, err)
		}
		dimensions[i] = int(dim)
	}

	// GGML Type
	ggmlType, err := g.readGGMLType(r)
	if err != nil {
		return GGUFTensorInfo{}, fmt.Errorf("failed to read ggml type: %w", err)
	}

	// Offset
	var offset uint64
	if err := binary.Read(r, byteOrder, &offset); err != nil {
		return GGUFTensorInfo{}, fmt.Errorf("failed to read offset: %w", err)
	}
	if offset%uint64(g.GetAlignment()) != 0 {
		return GGUFTensorInfo{}, fmt.Errorf("offset %d not multiple of alignment %d", offset, g.GetAlignment())
	}

	return GGUFTensorInfo{
		Name:       name,
		Dimensions: dimensions,
		GGMLType:   ggmlType,
		Offset:     offset,
	}, nil
}

func (g *GGUF) readGGMLType(r io.Reader) (tensor.GGMLType, error) {
	var typeID uint32
	if err := binary.Read(r, byteOrder, &typeID); err != nil {
		return tensor.GGMLType(0), err
	}
	return tensor.GGMLType(int(typeID)), nil
}

func (g *GGUF) readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, byteOrder, &length); err != nil {
		return "", fmt.Errorf("failed to read string length: %w", err)
	}

	if length == 0 {
		return "", nil
	}

	bytes := make([]byte, length)
	if _, err := io.ReadFull(r, bytes); err != nil {
		return "", fmt.Errorf("failed to read string bytes: %w", err)
	}

	return string(bytes), nil // Default Go string is UTF-8
}

func (g *GGUF) readKeyValuePair(r io.Reader) (key string, value interface{}, err error) {
	key, err = g.readString(r)
	if err != nil {
		return "", nil, fmt.Errorf("failed to read metadata key: %w", err)
	}

	// Check key constraints
	if len(key) >= (1 << 16) {
		return "", nil, fmt.Errorf("metadata key too long: %s", key)
	}
	for _, c := range key {
		if !(('a' <= c && c <= 'z') || ('0' <= c && c <= '9') || c == '_' || c == '.') {
			return "", nil, fmt.Errorf("metadata key contains invalid ASCII character: %s", key)
		}
	}

	value, err = g.readMetadataValue(r)
	if err != nil {
		return "", nil, fmt.Errorf("failed to read metadata value for key '%s': %w", key, err)
	}
	return key, value, nil
}

func (g *GGUF) readMetadataValue(r io.Reader) (interface{}, error) {
	valueType, err := g.readMetadataValueType(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read metadata value type: %w", err)
	}
	return g.readMetadataValueOfType(valueType, r)
}

func (g *GGUF) readMetadataValueType(r io.Reader) (MetadataValueType, error) {
	var index uint32
	if err := binary.Read(r, byteOrder, &index); err != nil {
		return 0, err
	}
	if index > uint32(MetadataValueTypeFLOAT64) {
		return 0, fmt.Errorf("invalid metadata value type index: %d", index)
	}
	return MetadataValueType(index), nil
}

func (g *GGUF) readMetadataValueOfType(valueType MetadataValueType, r io.Reader) (interface{}, error) {
	switch valueType {
	case MetadataValueTypeUINT8, MetadataValueTypeINT8:
		var val int8
		err := binary.Read(r, byteOrder, &val)
		return val, err
	case MetadataValueTypeUINT16, MetadataValueTypeINT16:
		var val int16
		err := binary.Read(r, byteOrder, &val)
		return val, err
	case MetadataValueTypeUINT32, MetadataValueTypeINT32:
		var val int32
		err := binary.Read(r, byteOrder, &val)
		return val, err
	case MetadataValueTypeFLOAT32:
		var val float32
		err := binary.Read(r, byteOrder, &val)
		return val, err
	case MetadataValueTypeUINT64, MetadataValueTypeINT64:
		var val int64
		err := binary.Read(r, byteOrder, &val)
		return val, err
	case MetadataValueTypeFLOAT64:
		var val float64
		err := binary.Read(r, byteOrder, &val)
		return val, err
	case MetadataValueTypeBOOL:
		var b byte
		if err := binary.Read(r, byteOrder, &b); err != nil {
			return false, err
		}
		return b != 0, nil
	case MetadataValueTypeSTRING:
		return g.readString(r)
	case MetadataValueTypeARRAY:
		return g.readArray(r)
	default:
		return nil, fmt.Errorf("unsupported metadata value type: %v", valueType)
	}
}

func (g *GGUF) readArray(r io.Reader) (interface{}, error) {
	valueType, err := g.readMetadataValueType(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read array element type: %w", err)
	}

	var length uint64
	if err := binary.Read(r, byteOrder, &length); err != nil {
		return nil, fmt.Errorf("failed to read array length: %w", err)
	}
	lenInt := int(length)

	// Read the elements into a slice of the appropriate type
	switch valueType {
	case MetadataValueTypeUINT8, MetadataValueTypeINT8:
		s := make([]byte, lenInt)
		if _, err := io.ReadFull(r, s); err != nil {
			return nil, err
		}
		return s, nil
	case MetadataValueTypeUINT16, MetadataValueTypeINT16:
		s := make([]int16, lenInt)
		err := binary.Read(r, byteOrder, s)
		return s, err
	case MetadataValueTypeUINT32, MetadataValueTypeINT32:
		s := make([]int32, lenInt)
		err := binary.Read(r, byteOrder, s)
		return s, err
	case MetadataValueTypeFLOAT32:
		s := make([]float32, lenInt)
		err := binary.Read(r, byteOrder, s)
		return s, err
	case MetadataValueTypeBOOL:
		s := make([]bool, lenInt)
		for i := 0; i < lenInt; i++ {
			b, err := g.readMetadataValueOfType(MetadataValueTypeBOOL, r)
			if err != nil {
				return nil, err
			}
			s[i] = b.(bool)
		}
		return s, nil
	case MetadataValueTypeSTRING:
		s := make([]string, lenInt)
		for i := 0; i < lenInt; i++ {
			str, err := g.readString(r)
			if err != nil {
				return nil, err
			}
			s[i] = str
		}
		return s, nil
	case MetadataValueTypeARRAY:
		s := make([]interface{}, lenInt)
		for i := 0; i < lenInt; i++ {
			arr, err := g.readArray(r) // Recursively call readArray
			if err != nil {
				return nil, err
			}
			s[i] = arr
		}
		return s, nil
	default:
		return nil, fmt.Errorf("unsupported array element type: %v", valueType)
	}
}

// GetAlignment returns the alignment value, defaulting to 32.
func (g *GGUF) GetAlignment() int {
	if g.Alignment != 0 {
		return g.Alignment
	}

	val, ok := g.Metadata["general.alignment"]
	if !ok {
		g.Alignment = DEFAULT_ALIGNMENT
	} else {

		if v, ok := val.(int32); ok {
			g.Alignment = int(v)
		} else if v, ok := val.(int64); ok {
			g.Alignment = int(v)
		} else if v, ok := val.(int8); ok {
			g.Alignment = int(v)
		} else {
			// Fallback if type assertion fails unexpectedly
			g.Alignment = DEFAULT_ALIGNMENT
		}
	}

	// Check if power of 2 (Go idiomatic way)
	if g.Alignment == 0 || (g.Alignment&(g.Alignment-1)) != 0 {

		g.Alignment = DEFAULT_ALIGNMENT
	}

	return g.Alignment
}
