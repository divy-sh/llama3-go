package llama

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"strings"

	"github.com/divy-sh/llama3-go/gguf"
	"github.com/divy-sh/llama3-go/model"
	"github.com/divy-sh/llama3-go/tensor"
	"github.com/divy-sh/llama3-go/tokenizer"
	"github.com/divy-sh/llama3-go/util"
)

const (
	TokenizerLlama3Model = "gpt2"
	// LLAMA_3_PATTERN is the regex used for Llama 3's tokenizer.

	LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)

// ModelLoader provides methods to load a Llama model from a GGUF file.
type ModelLoader struct{}

func loadVocabulary(metadata map[string]interface{}) (*tokenizer.Vocabulary, error) {
	model, ok := metadata["tokenizer.ggml.model"].(string)
	if !ok || model != TokenizerLlama3Model {
		return nil, fmt.Errorf("expected tokenizer.ggml.model '%s' but found '%v'", TokenizerLlama3Model, metadata["tokenizer.ggml.model"])
	}

	tokensRaw, ok := metadata["tokenizer.ggml.tokens"].([]string)
	if !ok {
		return nil, fmt.Errorf("missing or invalid 'tokenizer.ggml.tokens'")
	}

	// Assuming Vocabulary is a struct with a constructor
	return tokenizer.NewVocabulary(tokensRaw, nil), nil
}

// LoadModel opens the GGUF file and loads the model configuration and (optionally) weights.
func LoadModel(ggufPath string, contextLength int, loadWeights bool) (*Llama, error) {
	ggufFile, err := os.Open(ggufPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open GGUF file: %w", err)
	}
	defer ggufFile.Close()

	ggufData, err := gguf.LoadModel(ggufPath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse GGUF file: %w", err)
	}

	return LoadModelGguf(ggufFile, ggufData, contextLength, loadWeights)
}

func LoadModelGguf(r *os.File, ggufData *gguf.GGUF, contextLength int, shouldLoadWeights bool) (*Llama, error) {
	defer util.LogTimer("Load LlaMa model").Close()

	metadata := ggufData.Metadata
	vocabulary, err := loadVocabulary(metadata)
	if err != nil {
		return nil, err
	}
	tokenizer, err := createTokenizer(metadata, vocabulary)
	if err != nil {
		return nil, err
	}

	// Helper function for safe metadata access and type assertion
	getInt := func(key string) (int, error) {
		val, ok := metadata[key]
		if !ok {
			return 0, fmt.Errorf("missing metadata key: %s", key)
		}
		// Try int32 or int64 which are common GGUF types
		if v, ok := val.(int32); ok {
			return int(v), nil
		}
		if v, ok := val.(int64); ok {
			return int(v), nil
		}
		return 0, fmt.Errorf("invalid type for metadata key %s: got %T", key, val)
	}

	// Helper function for safe float metadata access
	getFloat := func(key string, defaultValue float32) float32 {
		if val, ok := metadata[key]; ok {
			if v, ok := val.(float32); ok {
				return v
			}
			if v, ok := val.(float64); ok {
				return float32(v)
			}
		}
		return defaultValue
	}

	dim, err := getInt("llama.embedding_length")
	if err != nil {
		return nil, err
	}
	hiddenDim, err := getInt("llama.feed_forward_length")
	if err != nil {
		return nil, err
	}
	numberOfLayers, err := getInt("llama.block_count")
	if err != nil {
		return nil, err
	}
	numberOfHeads, err := getInt("llama.attention.head_count")
	if err != nil {
		return nil, err
	}

	var numberOfKeyValueHeads int
	if kvHeads, err := getInt("llama.attention.head_count_kv"); err == nil {
		numberOfKeyValueHeads = kvHeads
	} else {
		numberOfKeyValueHeads = numberOfHeads // default to multi-head (no GQA)
	}

	config := NewConfiguration(
		dim,
		hiddenDim,
		numberOfLayers,
		numberOfHeads,
		numberOfKeyValueHeads,
		vocabulary.Size(),
		contextLength, // Use provided contextLength
		getFloat("llama.attention.layer_norm_rms_epsilon", 1e-5),
		getFloat("llama.rope.freq_base", 10000.0),
	).WithContextLength(contextLength)

	var weights *Weights
	if shouldLoadWeights {
		tensorEntries, err := gguf.LoadTensors(r.Name(), ggufData.GetTensorDataOffset(), ggufData.GetTensorInfos())
		if err != nil {
			return nil, fmt.Errorf("failed to load tensors: %w", err)
		}
		weights, err = LoadWeights(tensorEntries, config)
		if err != nil {
			return nil, fmt.Errorf("failed to process weights: %w", err)
		}
	}

	return NewLlama(config, tokenizer, weights), nil
}

func LoadWeights(tensorEntries map[string]tensor.GGMLTensorEntry, config *Configuration) (*Weights, error) {
	// RoPE Precomputation (assuming RoPE package/logic exists)
	ropeScaling := tensorEntries["rope_freqs"].Name != "" // check if 'rope_freqs' exists
	scaleFactor := float32(8.0)
	loFreqFactor := float32(1.0)
	hiFreqFactor := float32(3.0)
	oldContextLength := 8192

	ropeFreqsReal, ropeFreqsImag := model.RoPEPrecomputeFreqsCis(config.ContextLength, config.HeadSize, config.RopeTheta,
		ropeScaling, scaleFactor, loFreqFactor, hiFreqFactor, oldContextLength) // Assuming RoPEPrecomputeFreqsCis exists

	tokenEmbeddings, ok := tensorEntries["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("missing token_embd.weight tensor")
	}

	loadQuantized := func(entry tensor.GGMLTensorEntry) tensor.FloatTensor {
		return loadQuantizedTensor(entry) // Calls the function below
	}

	loadArrayOfQuantized := func(size int, getTensorEntry func(int) tensor.GGMLTensorEntry) []tensor.FloatTensor {
		array := make([]tensor.FloatTensor, size)
		for i := 0; i < size; i++ {
			array[i] = loadQuantized(getTensorEntry(i))
		}
		return array
	}

	loadArrayOfFloatBuffer := func(size int, getTensorEntry func(int) tensor.GGMLTensorEntry) [][]float32 {
		array := make([][]float32, size)
		for i := 0; i < size; i++ {
			entry := getTensorEntry(i)
			array[i] = toFloatBuffer(entry)
		}
		return array
	}

	getLayerTensor := func(i int, name string) tensor.GGMLTensorEntry {
		entry, ok := tensorEntries[fmt.Sprintf("blk.%d.%s", i, name)]
		if !ok {
			panic(fmt.Sprintf("Missing layer tensor blk.%d.%s", i, name))
		}
		return entry
	}

	// Weights for tied embeddings
	wclsEntry, ok := tensorEntries["output.weight"]
	if !ok {
		wclsEntry = tokenEmbeddings // Tie word embeddings
	}

	qw := &Weights{
		TokenEmbeddingTable: loadQuantized(tokenEmbeddings),
		RmsAttWeight:        loadArrayOfFloatBuffer(config.NumberOfLayers, func(i int) tensor.GGMLTensorEntry { return getLayerTensor(i, "attn_norm.weight") }),
		Wq:                  loadArrayOfQuantized(config.NumberOfLayers, func(i int) tensor.GGMLTensorEntry { return getLayerTensor(i, "attn_q.weight") }),
		Wk:                  loadArrayOfQuantized(config.NumberOfLayers, func(i int) tensor.GGMLTensorEntry { return getLayerTensor(i, "attn_k.weight") }),
		Wv:                  loadArrayOfQuantized(config.NumberOfLayers, func(i int) tensor.GGMLTensorEntry { return getLayerTensor(i, "attn_v.weight") }),
		Wo:                  loadArrayOfQuantized(config.NumberOfLayers, func(i int) tensor.GGMLTensorEntry { return getLayerTensor(i, "attn_output.weight") }),
		RmsFfnWeight:        loadArrayOfFloatBuffer(config.NumberOfLayers, func(i int) tensor.GGMLTensorEntry { return getLayerTensor(i, "ffn_norm.weight") }),
		W1:                  loadArrayOfQuantized(config.NumberOfLayers, func(i int) tensor.GGMLTensorEntry { return getLayerTensor(i, "ffn_gate.weight") }), // w1 (gate)
		W2:                  loadArrayOfQuantized(config.NumberOfLayers, func(i int) tensor.GGMLTensorEntry { return getLayerTensor(i, "ffn_down.weight") }), // w2 (down)
		W3:                  loadArrayOfQuantized(config.NumberOfLayers, func(i int) tensor.GGMLTensorEntry { return getLayerTensor(i, "ffn_up.weight") }),   // w3 (up)
		RmsFinalWeight:      toFloatBuffer(tensorEntries["output_norm.weight"]),
		FreqCisReal:         ropeFreqsReal,
		FreqCisImag:         ropeFreqsImag,
		Wcls:                loadQuantized(wclsEntry),
	}

	return qw, nil
}

func createTokenizer(metadata map[string]interface{}, vocabulary *tokenizer.Vocabulary) (*tokenizer.Tokenizer, error) {
	mergeLinesRaw, ok := metadata["tokenizer.ggml.merges"].([]string)
	if !ok {
		return nil, fmt.Errorf("missing or invalid 'tokenizer.ggml.merges'")
	}

	merges := make([]util.Pair[int, int], 0, len(mergeLinesRaw))
	for _, line := range mergeLinesRaw {
		parts := strings.Split(line, " ")
		if len(parts) != 2 {
			continue // Skip malformed lines
		}
		idx1, found1 := vocabulary.GetIndex(parts[0])
		idx2, found2 := vocabulary.GetIndex(parts[1])
		if !found1 || !found2 {
			// Throw equivalent in Go: log and maybe panic/return error
			return nil, fmt.Errorf("merge token not found in vocabulary: %s or %s", parts[0], parts[1])
		}
		merges = append(merges, util.Pair[int, int]{First: idx1, Second: idx2})
	}

	allTokens := vocabulary.Size()
	baseTokens := 128000 // Llama 3 standard base token count
	// reservedSpecialTokens := allTokens - baseTokens

	specialTokens := make(map[string]int)
	// Iterate through tokens from baseTokens up to allTokens
	for i := baseTokens; i < allTokens; i++ {
		tokenStr := vocabulary.Get(i)
		specialTokens[tokenStr] = i
	}

	// Assuming Tokenizer is a struct with a constructor
	return tokenizer.NewTokenizer(vocabulary, merges, LLAMA_3_PATTERN, specialTokens), nil
}

// loadQuantizedTensor loads a tensor entry and returns a FloatTensor based on its GGML type.
// (Assuming FloatTensor, Q8_0FloatTensor, etc., are interfaces/structs in the `llama` package)
func loadQuantizedTensor(entry tensor.GGMLTensorEntry) tensor.FloatTensor {
	elements := FloatTensorNumberOfElements(entry.Shape) // Assuming helper exists

	switch entry.GGMLType {
	// case tensor.F32:
	// return tensor.NewF32FloatTensor(elements, entry.Data)
	case tensor.Q8_0:
		return tensor.NewQ8_0FloatTensor(elements, entry.Data)
	case tensor.Q4_0:
		return tensor.NewQ4_0FloatTensor(elements, entry.Data)
	case tensor.BF16:
		return tensor.NewBF16FloatTensor(elements, entry.Data)
	case tensor.F16:
		return tensor.NewF16FloatTensor(elements, entry.Data)
	default:
		panic(fmt.Sprintf("Quantization format %v not supported", entry.GGMLType))
	}
}

// toFloatBuffer converts a tensor entry of type F32 into a Go []float32 slice.
func toFloatBuffer(tensorEntry tensor.GGMLTensorEntry) []float32 {
	if tensorEntry.GGMLType != tensor.F32 {
		panic(fmt.Sprintf("Conversion to F32 slice unsupported for type %v", tensorEntry.GGMLType))
	}

	// We assume entry.Data is a []byte or similar segment of the tensor data.
	byteData := tensorEntry.Data.Bytes() // Assuming entry.Data provides access to bytes

	numFloats := len(byteData) / 4
	floats := make([]float32, numFloats)

	// GGUF is Little Endian
	buf := bytes.NewReader(byteData)
	err := binary.Read(buf, binary.LittleEndian, &floats)
	if err != nil {
		panic(fmt.Sprintf("Error reading F32 data: %v", err))
	}

	return floats
}
