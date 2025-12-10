package aot

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/divy-sh/llama3-go/gguf"
	"github.com/divy-sh/llama3-go/llama"
)

type PartialModel struct {
	ModelFileName    string
	Model            *llama.Llama
	TensorDataOffset int64
	TensorInfos      map[string]gguf.GGUFTensorInfo
}

var PRELOADED_GGUF *PartialModel

func init() {
	PRELOADED_GGUF = preLoadGGUF(os.Getenv("LLAMA_PRELOAD_GGUF"))
}

func preLoadGGUF(modelPath string) *PartialModel {
	if modelPath == "" {
		return nil
	}

	path := modelPath

	if _, err := os.Stat(path); os.IsNotExist(err) {
		panic(fmt.Sprintf("Cannot pre-load model: file not found at %s", path))
	}

	gguf, err := llama.LoadModel(path)
	if err != nil {
		panic(fmt.Sprintf("Failed to load GGUF metadata from %s: %v", path, err))
	}

	file, err := os.Open(path)
	if err != nil {
		panic(fmt.Sprintf("Failed to open model file %s: %v", path, err))
	}
	defer file.Close()

	const defaultMaxTokens = 8192

	ml := &ModelLoader{}

	baseModel, err := ml.LoadModel(file, gguf, defaultMaxTokens, false)
	if err != nil {
		panic(fmt.Sprintf("Failed to load base model structure from %s: %v", path, err))
	}

	return &PartialModel{
		ModelFileName:    filepath.Base(path),
		Model:            baseModel,
		TensorDataOffset: gguf.GetTensorDataOffset(),
		TensorInfos:      gguf.GetTensorInfos(),
	}
}

func TryUsePreLoaded(modelPath string, contextLength int) (*llama.Llama, error) {
	preLoaded := PRELOADED_GGUF
	if preLoaded == nil {
		return nil, nil
	}

	optionsModel := filepath.Base(modelPath)
	preLoadedModel := preLoaded.ModelFileName

	if !strings.EqualFold(optionsModel, preLoadedModel) {
		return nil, nil
	}

	baseModel := preLoaded.Model

	// timer := Log("Load tensors from pre-loaded model")
	// defer timer.Close()

	file, err := os.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file for tensor loading: %w", err)
	}
	defer file.Close()

	tensorEntries, err := LoadTensors(file, preLoaded.TensorDataOffset, preLoaded.TensorInfos)
	if err != nil {
		return nil, fmt.Errorf("failed to load tensors from preloaded model file: %w", err)
	}

	// Load weights using the tensor entries
	ml := &ModelLoader{}
	weights, err := ml.LoadWeights(tensorEntries, baseModel.Configuration())
	if err != nil {
		return nil, fmt.Errorf("failed to load weights from tensor entries: %w", err)
	}

	// Create a new Llama model instance with the specified context length.
	newConfig := baseModel.Configuration().WithContextLength(contextLength)

	return &Llama{
		Config:    newConfig,
		Tokenizer: baseModel.GetTokenizer(),
		Weights:   weights,
	}, nil
}
