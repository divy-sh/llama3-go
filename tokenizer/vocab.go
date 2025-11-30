package tokenizer

type Vocabulary struct {
	Tokens       []string
	Scores       []float32
	TokenToIndex map[string]int
}

// NewVocabulary is the constructor equivalent
func NewVocabulary(tokens *[]string, scores *[]float32) *Vocabulary {
	tokenToIndex := make(map[string]int)
	for i, token := range *tokens {
		tokenToIndex[token] = i
	}
	return &Vocabulary{
		Tokens:       *tokens,
		Scores:       *scores,
		TokenToIndex: tokenToIndex,
	}
}

// Get returns the token at the given index
func (v *Vocabulary) Get(index int) string {
	if index < 0 || index >= len(v.Tokens) {
		return "" // or handle error
	}
	return v.Tokens[index]
}

// GetIndex returns the index of a token, or -1 if not found
func (v *Vocabulary) GetIndex(token string) int {
	if index, ok := v.TokenToIndex[token]; ok {
		return index
	}
	return -1 // optional: could return (int, bool) for more idiomatic Go
}

// Size returns the number of tokens
func (v *Vocabulary) Size() int {
	return len(v.Tokens)
}
