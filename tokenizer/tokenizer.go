package tokenizer

import (
	"regexp"
)

type Pair [2]int

type Tokenizer struct {
	pattern       regexp.Regexp
	vocab         Vocabulary
	merges        map[Pair]int
	specialTokens map[string]int
}

func NewTokenizer(vocab Vocabulary, merges map[Pair]int, specialTokens map[string]int, pattern string) *Tokenizer {
	compiledPattern := regexp.MustCompile(pattern)
	return &Tokenizer{
		pattern:       *compiledPattern,
		vocab:         vocab,
		merges:        merges,
		specialTokens: specialTokens,
	}
}

func (t *Tokenizer) RegexPattern() *string {
	if t.pattern.String() == "" {
		return nil
	}
	patternString := t.pattern.String()
	return &patternString
}

func (t *Tokenizer) GetSpecialTokens() map[string]int {
	return t.specialTokens
}

func (t *Tokenizer) isSpecialToken() bool {
	return len(t.specialTokens) > 0
}

func (t *Tokenizer) EncodeAsList(name string) []int {
	panic("unimplemented")
}

func encodeImpl(name string) []int {
	panic("unimplemented")
}
