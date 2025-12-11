package tokenizer

import (
	"fmt"
	"math"
	"regexp"
	"strings"
	"sync"

	"github.com/divy-sh/llama3-go/util"
)

// Tokenizer implements the Byte Pair Encoding algorithm.
type Tokenizer struct {
	compiledPattern *regexp.Regexp
	vocabulary      *Vocabulary                 // Vocabulary holds the mapping from token ID to string
	merges          map[util.Pair[int, int]]int // BPE merges: (tokenA, tokenB) -> mergedTokenID
	specialTokens   map[string]int              // Map of special token string to token ID
}

// NewTokenizer creates a new BPE Tokenizer.
// The BPE merges are pre-processed to map (token1_id, token2_id) -> merged_token_id.
func NewTokenizer(vocabulary *Vocabulary, merges []util.Pair[int, int], regexPattern string, specialTokens map[string]int) *Tokenizer {
	t := &Tokenizer{
		vocabulary:    vocabulary,
		specialTokens: make(map[string]int, len(specialTokens)),
		merges:        make(map[util.Pair[int, int]]int),
	}

	for k, v := range specialTokens {
		t.specialTokens[k] = v
	}

	if regexPattern != "" {
		t.compiledPattern = regexp.MustCompile(regexPattern)
	}

	for _, pair := range merges {
		tokenStr1 := vocabulary.Get(pair.First)
		tokenStr2 := vocabulary.Get(pair.Second)

		mergedTokenStr := tokenStr1 + tokenStr2

		// This must exist in the vocabulary as it was constructed during BPE training
		mergeIndex, found := vocabulary.GetIndex(mergedTokenStr)
		if !found {
			// This is a setup error; tokenizers rely on the vocabulary being complete.
			panic(fmt.Sprintf("internal merge token %q not found in vocabulary", mergedTokenStr))
		}
		t.merges[pair] = mergeIndex
	}

	return t
}

// RegexPattern returns the regex pattern string used for splitting text.
func (t *Tokenizer) RegexPattern() string {
	if t.compiledPattern == nil {
		return ""
	}
	return t.compiledPattern.String()
}

// GetSpecialTokens returns the map of special token strings to IDs.
func (t *Tokenizer) GetSpecialTokens() map[string]int {
	return t.specialTokens
}

// IsSpecialToken checks if a token ID corresponds to a special token.
func (t *Tokenizer) IsSpecialToken(tokenIndex int) bool {
	for _, id := range t.specialTokens {
		if id == tokenIndex {
			return true
		}
	}
	return false
}

// EncodeOrdinary encodes a string without handling special tokens.
func (t *Tokenizer) EncodeOrdinary(text string) []int {
	// Split text into chunks using the regex pattern
	if t.compiledPattern == nil {
		return []int{} // Cannot tokenize if pattern is missing
	}
	textChunks := t.compiledPattern.FindAllString(text, -1)

	// Encode each chunk and combine results
	ids := make([]int, 0)
	for _, chunk := range textChunks {
		ids = append(ids, t.encodeChunk(chunk)...)
	}
	return ids
}

// encodeChunk tokenizes a single chunk of text using BPE merging.
func (t *Tokenizer) encodeChunk(chunk string) []int {
	// Convert initial chunk to a list of base token IDs (single character/byte tokens)
	// In the original BPE, this happens after byte-to-unicode mapping (See Encode method).

	runes := []rune(chunk)
	ids := make([]int, 0, len(runes))
	for _, r := range runes {
		// Convert rune back to string to look up in the vocabulary
		tokenStr := string(r)
		tokenIndex, found := t.vocabulary.GetIndex(tokenStr)
		if !found {
			// Should not happen in a correctly set up BPE tokenizer
			panic(fmt.Sprintf("base token %q not found in vocabulary", tokenStr))
		}
		ids = append(ids, tokenIndex)
	}

	// Repeatedly apply the best BPE merge
	for len(ids) >= 2 {
		// Find the best pair to merge (the one with the lowest merged token ID)
		stats := t.getStats(ids)

		bestPair := util.Pair[int, int]{}
		minMergeID := math.MaxInt
		foundMerge := false

		// Find the pair with the smallest merge ID that is present in `t.merges`
		for pair := range stats {
			if mergeID, ok := t.merges[pair]; ok {
				if mergeID < minMergeID {
					minMergeID = mergeID
					bestPair = pair
					foundMerge = true
				}
			}
		}

		if !foundMerge {
			break // Nothing else can be merged anymore
		}

		// Merge the best pair
		ids = t.merge(ids, bestPair, minMergeID)
	}
	return ids
}

// getStats counts the frequency of adjacent token pairs in the IDs list.
func (t *Tokenizer) getStats(ids []int) map[util.Pair[int, int]]int {
	stats := make(map[util.Pair[int, int]]int)
	for i := 0; i+1 < len(ids); i++ {
		pair := util.Pair[int, int]{First: ids[i], Second: ids[i+1]}
		stats[pair]++
	}
	return stats
}

// merge applies a single BPE merge operation to the list of IDs.
func (t *Tokenizer) merge(ids []int, pair util.Pair[int, int], idx int) []int {
	newIDs := make([]int, 0, len(ids))
	i := 0
	for i < len(ids) {
		// If not at the very last position AND the pair matches, replace it
		if i < len(ids)-1 && ids[i] == pair.First && ids[i+1] == pair.Second {
			newIDs = append(newIDs, idx)
			i += 2
		} else {
			newIDs = append(newIDs, ids[i])
			i += 1
		}
	}
	return newIDs
}

// DecodeImpl takes a list of token IDs and concatenates their string representations.
func (t *Tokenizer) DecodeImpl(tokens []int) string {
	var sb strings.Builder
	for _, token := range tokens {
		tokenString := t.vocabulary.Get(token)
		sb.WriteString(tokenString)
	}
	return sb.String()
}

// Encode handles special tokens by splitting the text.
func (t *Tokenizer) Encode(text string, allowedSpecial map[string]struct{}) []int {
	if len(allowedSpecial) == 0 {
		// Shortcut: if no special tokens are allowed/present, just use the ordinary encoding
		return t.EncodeOrdinary(text)
	}

	// Build a regex pattern to split text by special tokens
	var patternParts []string
	for token := range allowedSpecial {
		patternParts = append(patternParts, regexp.QuoteMeta(token))
	}
	// The pattern surrounding with () makes it a capturing group, so the special tokens are included in the split result.
	specialPattern := "(" + strings.Join(patternParts, "|") + ")"

	re := regexp.MustCompile(specialPattern)

	// Split the text, including the delimiters (special tokens)
	chunks := re.FindAllStringIndex(text, -1)
	ids := make([]int, 0)
	lastIndex := 0

	for _, chunkIndices := range chunks {
		start := chunkIndices[0]
		end := chunkIndices[1]

		// Ordinary chunk between last match and current match
		if start > lastIndex {
			ordinaryChunk := text[lastIndex:start]
			ids = append(ids, t.EncodeOrdinary(ordinaryChunk)...)
		}

		// Special token match
		specialToken := text[start:end]
		if id, ok := t.specialTokens[specialToken]; ok {
			ids = append(ids, id)
		} else {
			// This path should ideally not be taken if allowedSpecial is a subset of specialTokens
			// and the regex split only captured special tokens.
			ids = append(ids, t.EncodeOrdinary(specialToken)...)
		}

		lastIndex = end
	}

	// Remainder of the text
	if lastIndex < len(text) {
		ordinaryChunk := text[lastIndex:]
		ids = append(ids, t.EncodeOrdinary(ordinaryChunk)...)
	}

	return ids
}

// Byte-to-Unicode Mapping for BPE Consistency (GPT-2 Style)

var (
	byteEncoder map[int]rune
	byteDecoder map[rune]int
	once        sync.Once
)

func bytesToUnicode() (map[int]rune, map[rune]int) {
	bs := make([]int, 0, 256)  // Byte values 0-255
	cs := make([]rune, 0, 256) // Corresponding unicode code points

	// Printable ASCII
	for b := int('!'); b <= int('~'); b++ {
		bs = append(bs, b)
	}
	// Latin-1 Supplement (partial)
	for b := 0xA1; b <= 0xAC; b++ { // ¡ to ¬
		bs = append(bs, b)
	}
	// Latin-1 Supplement (partial)
	for b := 0xAE; b <= 0xFF; b++ { // ® to ÿ
		bs = append(bs, b)
	}

	// Map to initial corresponding code points
	for _, b := range bs {
		cs = append(cs, rune(b))
	}

	// Map remaining 0-255 bytes to the unused Unicode range 256 onwards
	n := 0
	for b := 0; b < 256; b++ {
		found := false
		for _, existingB := range bs {
			if existingB == b {
				found = true
				break
			}
		}
		if !found {
			bs = append(bs, b)
			cs = append(cs, rune(256+n))
			n++
		}
	}

	encoder := make(map[int]rune)
	decoder := make(map[rune]int)
	for i := 0; i < len(bs); i++ {
		encoder[bs[i]] = cs[i]
		decoder[cs[i]] = bs[i]
	}

	return encoder, decoder
}

func initByteMappings() {
	once.Do(func() {
		byteEncoder, byteDecoder = bytesToUnicode()
	})
}

// Encode converts the raw string to token IDs.
func (t *Tokenizer) EncodeString(text string) []int {
	initByteMappings()

	// Convert UTF-8 bytes to the 'clean' Unicode code points for BPE
	// This ensures that the BPE algorithm works on a consistent set of 256 'characters'.
	var sb strings.Builder
	rawBytes := []byte(text)
	for _, b := range rawBytes {
		sb.WriteRune(byteEncoder[int(b)])
	}

	// Perform BPE encoding on the mapped string (without special token handling here)

	return t.EncodeOrdinary(sb.String())
}

// Decode converts token IDs back to a raw string.
func (t *Tokenizer) Decode(tokens []int) string {
	initByteMappings()

	// Decode IDs to the mapped Unicode string
	mappedString := t.DecodeImpl(tokens)

	// Convert the mapped Unicode code points back to raw UTF-8 bytes
	var rawBytes []byte
	for _, r := range mappedString {
		byteVal := byteDecoder[r]
		rawBytes = append(rawBytes, byte(byteVal))
	}

	// Convert bytes back to a UTF-8 string
	return string(rawBytes)
}

// ReplaceControlCharacters replaces control characters (except newline) with their escaped representation.
func ReplaceControlCharacters(str string) string {
	var sb strings.Builder
	for _, r := range str {
		if r != '\n' && (r < ' ' || (r >= 0x7F && r <= 0x9F) || (r >= 0x2000 && r <= 0x2FFF)) {
			// Basic check for control characters. A more complete check would use unicode.IsControl

			sb.WriteString(fmt.Sprintf("\\u%04x", r))
		} else {
			sb.WriteRune(r)
		}
	}
	return sb.String()
}
