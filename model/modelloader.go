package model

import (
	"errors"

	"github.com/divy-sh/llama3-go/tokenizer"
)

const (
	tokenizerLlama3Model = "gp2"
	llama3Pattern        = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)

type ModelLoader struct {
}

func LoadVocab(metadata map[string]interface{}) (*tokenizer.Vocabulary, error) {
	model := metadata["tokenizer.ggml.model"].(string)
	if tokenizerLlama3Model != model {
		return nil, errors.New("unsupported tokenizer model: " + model)
	}

	tokens := metadata["tokenizer.ggml.tokens"].([]string)
	return tokenizer.NewVocabulary(&tokens, nil), nil
}

func LoadModel(ggufPath string, contextLength int, loadWeights bool) (*Llama, error) {
	gguf := gguf.LoadModel(ggufPath)
}
