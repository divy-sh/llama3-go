package tokenizer

import (
	"strconv"
)

type Vocabulary struct {
	Tokens       []string
	Scores       []float32
	TokenToIndex map[string]int
}

// NewVocabulary creates a new Vocabulary instance, building the lookup map.
func NewVocabulary(tokens []string, scores []float32) *Vocabulary {
	tokenToIndex := make(map[string]int, len(tokens))
	for i, token := range tokens {
		tokenToIndex[token] = i
	}
	return &Vocabulary{
		Tokens:       tokens,
		Scores:       scores,
		TokenToIndex: tokenToIndex,
	}
}

// Get returns the token string for a given index.
func (v *Vocabulary) Get(tokenIndex int) string {
	if tokenIndex < 0 || tokenIndex >= len(v.Tokens) {
		return "[Unknown Token Index: " + strconv.Itoa(tokenIndex) + "]"
	}
	return v.Tokens[tokenIndex]
}

// GetIndex returns the index of a token, or -1 if not found (simulating OptionalInt).
func (v *Vocabulary) GetIndex(token string) (int, bool) {
	if index, ok := v.TokenToIndex[token]; ok {
		return index, true
	}
	return -1, false // Sentinel for not found, similar to OptionalInt.empty()
}

// Size returns the total number of tokens in the vocabulary.
func (v *Vocabulary) Size() int {
	return len(v.Tokens)
}
