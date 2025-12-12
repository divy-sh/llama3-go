package llama

import (
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	"github.com/divy-sh/llama3-go/tensor"
	"github.com/divy-sh/llama3-go/tokenizer"
	"github.com/divy-sh/llama3-go/util"
)

// Llama struct encapsulates the model configuration, tokenizer, and weights.

type Llama struct {
	Configuration *Configuration
	Tokenizer     *tokenizer.Tokenizer
	Weights       *Weights
}

func NewLlama(config *Configuration, tokenizer *tokenizer.Tokenizer, weights *Weights) *Llama {
	return &Llama{
		Configuration: config,
		Tokenizer:     tokenizer,
		Weights:       weights,
	}
}

// CreateNewState allocates a new State structure for the forward pass.
func (l *Llama) CreateNewState(batchSize int) *State {
	state := NewState(l.Configuration, batchSize)

	if botToken, ok := l.Tokenizer.GetSpecialTokens()["<|begin_of_text|>"]; ok {
		state.LatestToken = botToken
	} else {
		// Fallback for models without this token
		state.LatestToken = 1 // Common default for BOS/first token
	}
	return state
}

// Configuration struct holds the model hyperparameters.
type Configuration struct {
	Dim                   int // transformer dimension
	HiddenDim             int // for ffn layers
	NumberOfLayers        int // number of layers
	NumberOfHeads         int // number of query heads
	NumberOfKeyValueHeads int // number of key/value heads
	VocabularySize        int // vocabulary size
	ContextLength         int // max sequence length
	RmsNormEps            float32
	RopeTheta             float32
	HeadSize              int
}

func NewConfiguration(dim, hiddenDim, numberOfLayers, numberOfHeads, numberOfKeyValueHeads, vocabularySize, contextLength int, rmsNormEps, ropeTheta float32) *Configuration {
	return &Configuration{
		Dim:                   dim,
		HiddenDim:             hiddenDim,
		NumberOfLayers:        numberOfLayers,
		NumberOfHeads:         numberOfHeads,
		NumberOfKeyValueHeads: numberOfKeyValueHeads,
		VocabularySize:        vocabularySize,
		ContextLength:         contextLength,
		RmsNormEps:            rmsNormEps,
		RopeTheta:             ropeTheta,
		HeadSize:              dim / numberOfHeads,
	}
}

func (c *Configuration) WithContextLength(newContextLength int) *Configuration {
	if newContextLength < 0 {
		return c
	}
	// Return a copy with the new context length
	return NewConfiguration(
		c.Dim, c.HiddenDim, c.NumberOfLayers, c.NumberOfHeads, c.NumberOfKeyValueHeads,
		c.VocabularySize, newContextLength, c.RmsNormEps, c.RopeTheta,
	)
}

// Weights struct holds all the model weights, stored as FloatTensor or []float32 slices.
type Weights struct {
	TokenEmbeddingTable tensor.FloatTensor // (vocab_size, dim)
	RmsAttWeight        [][]float32        // (layer, dim) - using []float32 slice for FloatBuffer equivalent
	Wq                  []tensor.FloatTensor
	Wk                  []tensor.FloatTensor
	Wv                  []tensor.FloatTensor
	Wo                  []tensor.FloatTensor
	RmsFfnWeight        [][]float32
	W1                  []tensor.FloatTensor // gate
	W2                  []tensor.FloatTensor // down
	W3                  []tensor.FloatTensor // up
	RmsFinalWeight      []float32            // (dim,)
	FreqCisReal         []float32            // RoPE freqs
	FreqCisImag         []float32            // RoPE freqs
	Wcls                tensor.FloatTensor   // (vocab_size, dim)
}

// State struct holds the activations and the KV cache.
type State struct {
	BatchSize int
	X         []tensor.FloatTensor // activation at current time stamp (dim,)
	Xb        []tensor.FloatTensor // same, but inside a residual branch (dim,)
	Xb2       []tensor.FloatTensor // an additional buffer (dim,)
	Hb        []tensor.FloatTensor // buffer for hidden dimension in the ffn (hidden_dim,)
	Hb2       []tensor.FloatTensor // buffer for hidden dimension in the ffn (hidden_dim,)
	Q         []tensor.FloatTensor // query (dim,)
	K         []tensor.FloatTensor // key (dim,)
	V         []tensor.FloatTensor // value (dim,)
	Att       []tensor.FloatTensor // buffer for scores/attention values (n_heads, seq_len)
	Logits    tensor.FloatTensor   // output logits

	KeyCache   []tensor.FloatTensor // (n_layer, seq_len, kv_dim)
	ValueCache []tensor.FloatTensor // (n_layer, seq_len, kv_dim)

	IdxPrevBlock int // last index in previous block
	LatestToken  int
}

// NewState allocates and initializes the state buffers.
func NewState(config *Configuration, batchSize int) *State {
	kvDim := (config.Dim * config.NumberOfKeyValueHeads) / config.NumberOfHeads

	// Helper function to allocate FloatTensor array for a given shape
	allocate := func(numTokens int, dims ...int) []tensor.FloatTensor {
		tensors := make([]tensor.FloatTensor, numTokens)
		for i := range tensors {
			tensors[i] = tensor.ArrayFloatTensorAllocate(dims...)
		}
		return tensors
	}

	state := &State{
		BatchSize:    batchSize,
		X:            allocate(batchSize, config.Dim),
		Xb:           allocate(batchSize, config.Dim),
		Xb2:          allocate(batchSize, config.Dim),
		Hb:           allocate(batchSize, config.HiddenDim),
		Hb2:          allocate(batchSize, config.HiddenDim),
		Q:            allocate(batchSize, config.Dim),
		K:            allocate(batchSize, config.Dim),
		V:            allocate(batchSize, config.Dim),
		Att:          allocate(batchSize, config.NumberOfHeads, config.ContextLength),
		IdxPrevBlock: -1,
		Logits:       tensor.ArrayFloatTensorAllocate(config.VocabularySize),
		KeyCache:     make([]tensor.FloatTensor, config.NumberOfLayers),
		ValueCache:   make([]tensor.FloatTensor, config.NumberOfLayers),
	}

	// Allocate KV Cache
	for l := 0; l < config.NumberOfLayers; l++ {
		state.KeyCache[l] = tensor.ArrayFloatTensorAllocate(config.ContextLength, kvDim)
		state.ValueCache[l] = tensor.ArrayFloatTensorAllocate(config.ContextLength, kvDim)
	}

	return state
}

// rmsnorm computes RMS normalization.
func rmsnorm(out, x tensor.FloatTensor, weight []float32, size int, rmsNormEps float32) {
	// calculate sum of squares
	ss := float32(0.0)
	for i := 0; i < size; i++ {
		xi := x.GetFloat(i)
		ss += xi * xi
	}
	ss /= float32(size)
	ss += rmsNormEps
	ss = 1.0 / float32(math.Sqrt(float64(ss)))

	// normalize and scale
	for i := 0; i < size; i++ {
		val := weight[i] * (ss * x.GetFloat(i))
		out.SetFloat(i, val)
	}
}

// forward runs the model's forward pass for a chunk of tokens (batch size).
func Forward(model *Llama, state *State, tokens []int, position int, computeLogits bool) tensor.FloatTensor {
	config := model.Configuration
	weights := model.Weights
	dim := config.Dim
	headSize := config.HeadSize
	kvDim := (config.Dim * config.NumberOfKeyValueHeads) / config.NumberOfHeads
	kvMul := config.NumberOfHeads / config.NumberOfKeyValueHeads
	sqrtHeadSize := float32(math.Sqrt(float64(headSize)))
	nTokens := len(tokens)

	// 1. Token Embedding
	// Parallel copy the token embeddings into x
	util.ParallelFor(0, nTokens, func(t int) {
		weights.TokenEmbeddingTable.CopyTo(tokens[t]*dim, state.X[t], 0, dim)
	})

	// 2. Transformer Layers
	for l := 0; l < config.NumberOfLayers; l++ {
		// a. Attention RMSNorm
		curLayer := l
		util.ParallelFor(0, nTokens, func(t int) {
			rmsnorm(state.Xb[t], state.X[t], weights.RmsAttWeight[curLayer], dim, config.RmsNormEps)
		})

		// b. QKV Matmuls
		weights.Wq[l].Matmul(nTokens, state.Xb, state.Q, dim, dim)
		weights.Wk[l].Matmul(nTokens, state.Xb, state.K, kvDim, dim)
		weights.Wv[l].Matmul(nTokens, state.Xb, state.V, kvDim, dim)

		// c. RoPE Relative Positional Encoding
		util.ParallelFor(0, nTokens, func(t int) {
			for i := 0; i < dim; i += 2 {
				headDim := i % headSize

				// Calculate RoPE indices
				freqRealIndex := (position+t)*(headSize/2) + (headDim / 2)
				freqImagIndex := freqRealIndex

				fcr := weights.FreqCisReal[freqRealIndex]
				fci := weights.FreqCisImag[freqImagIndex]
				rotn := 1
				if i < kvDim {
					rotn = 2 // Rotate Q and K
				} else {
					rotn = 1 // Rotate Q only (in the full dimension space, K is smaller/shared)
				}

				for vi := 0; vi < rotn; vi++ {
					var vec tensor.FloatTensor
					if vi == 0 {
						vec = state.Q[t] // Query
					} else {
						vec = state.K[t] // Key
					}
					v0 := vec.GetFloat(i)
					v1 := vec.GetFloat(i + 1)

					// Complex multiplication: (v0 + i*v1) * (fcr + i*fci) = (v0*fcr - v1*fci) + i*(v0*fci + v1*fcr)
					vec.SetFloat(i, v0*fcr-v1*fci)
					vec.SetFloat(i+1, v0*fci+v1*fcr)
				}
			}
		})

		// d. Save to KV Cache
		util.ParallelFor(0, nTokens, func(t int) {
			// loff is implicit in KeyCache[curLayer]
			keyCacheOffset := (position + t) * kvDim
			state.K[t].CopyTo(0, state.KeyCache[curLayer], keyCacheOffset, kvDim)
			state.V[t].CopyTo(0, state.ValueCache[curLayer], keyCacheOffset, kvDim)
		})

		// If logits are not required, skip attention and FFN on the last layer
		if !computeLogits && curLayer == config.NumberOfLayers-1 {
			state.IdxPrevBlock = nTokens - 1
			return nil
		}

		// e. Multihead Attention
		var wg sync.WaitGroup
		wg.Add(nTokens * config.NumberOfHeads)

		for t := 0; t < nTokens; t++ {
			token := t
			for h := 0; h < config.NumberOfHeads; h++ {
				head := h
				go func() {
					defer wg.Done()

					qOffset := head * headSize
					attOffset := head * config.ContextLength

					// 1. Calculate Attention Scores (Dot Product)
					for tStep := 0; tStep <= position+token; tStep++ {
						// keyCacheOffset: (t * kvDim) + (h/kvMul * headSize)
						keyCacheOffset := tStep*kvDim + (head/kvMul)*headSize

						score := state.Q[token].Dot(qOffset, state.KeyCache[curLayer], keyCacheOffset, headSize)
						score /= sqrtHeadSize
						state.Att[token].SetFloat(attOffset+tStep, score)
					}

					// 2. Softmax Scores
					state.Att[token].SoftmaxInPlace(attOffset, position+token+1)

					// 3. Weighted Sum of Values
					xbOffset := head * headSize
					state.Xb[token].FillInPlace(xbOffset, headSize, 0.0) // memset/FillInPlace

					for tStep := 0; tStep <= position+token; tStep++ {
						// vOffset: (t * kvDim) + (h/kvMul * headSize)
						vOffset := tStep*kvDim + (head/kvMul)*headSize

						attentionWeight := state.Att[token].GetFloat(attOffset + tStep)
						// accumulate the weighted value into xb
						state.Xb[token].SaxpyInPlace(xbOffset, state.ValueCache[curLayer], vOffset, headSize, attentionWeight)
					}
				}()
			}
		}
		wg.Wait()

		// f. Output Matmul
		weights.Wo[l].Matmul(nTokens, state.Xb, state.Xb2, dim, dim)

		// g. Residual Connection (Attention)
		util.ParallelFor(0, nTokens, func(t int) {
			state.X[t].AddInPlace(state.Xb2[t])
		})

		// h. FFN RMSNorm
		util.ParallelFor(0, nTokens, func(t int) {
			rmsnorm(state.Xb[t], state.X[t], weights.RmsFfnWeight[curLayer], dim, config.RmsNormEps)
		})

		// i. FFN Matmuls (w1, w3)
		weights.W1[l].Matmul(nTokens, state.Xb, state.Hb, config.HiddenDim, dim)  // w1 (gate)
		weights.W3[l].Matmul(nTokens, state.Xb, state.Hb2, config.HiddenDim, dim) // w3 (up)

		// j. SwiGLU Non-linearity
		util.ParallelFor(0, nTokens, func(t int) {
			// silu(x) = x / (1 + exp(-x)) -> applied to Hb
			state.Hb[t].MapInPlace(func(value float32) float32 {
				return value / float32(1.0+math.Exp(float64(-value)))
			})
		})

		// k. Element-wise Multiply (gate * up)
		util.ParallelFor(0, nTokens, func(t int) {
			state.Hb[t].MultiplyInPlace(state.Hb2[t])
		})

		// l. FFN Final Matmul
		weights.W2[l].Matmul(nTokens, state.Hb, state.Xb, dim, config.HiddenDim) // w2 (down)

		// m. Residual Connection (FFN)
		util.ParallelFor(0, nTokens, func(t int) {
			state.X[t].AddInPlace(state.Xb[t])
		})
	}

	// 3. Final RMSNorm
	util.ParallelFor(0, nTokens, func(t int) {
		rmsnorm(state.X[t], state.X[t], weights.RmsFinalWeight, dim, config.RmsNormEps)
	})

	// 4. Classifier into logits
	// Only the last token in the batch is used for next token prediction
	weights.Wcls.Matmul(1, []tensor.FloatTensor{state.X[nTokens-1]}, []tensor.FloatTensor{state.Logits}, config.VocabularySize, dim)
	state.IdxPrevBlock = nTokens - 1

	return state.Logits
}

// GenerateTokens implements the main LLM generation loop.
func GenerateTokens(model *Llama, state *State, startPosition int, promptTokens []int, stopTokens map[int]struct{}, maxTokens int, sampler Sampler, echo bool, onTokenGenerated func(int)) []int {
	startNanos := time.Now().UnixNano()
	startGen := int64(0)

	if maxTokens < 0 || model.Configuration.ContextLength < maxTokens {
		maxTokens = model.Configuration.ContextLength
	}

	generatedTokens := make([]int, 0, maxTokens)
	token := state.LatestToken // BOS?
	var nextToken int
	promptIndex := 0

	for position := startPosition; position < maxTokens; position++ {
		if promptIndex < len(promptTokens) {
			// --- Prompt Ingestion (Batch Processing) ---

			// Calculate batch size (nTokens)
			nTokens := min(maxTokens-position, min(len(promptTokens)-promptIndex, state.BatchSize))
			tokens := make([]int, nTokens)

			for i := 0; i < nTokens; i++ {
				tokens[i] = promptTokens[promptIndex+i]
				if echo {
					// log prompt token
					fmt.Fprint(os.Stderr, TokenizerReplaceControlCharacters(model.Tokenizer.Decode([]int{tokens[i]})))
				}
			}

			// Only compute logits on the very last batch for efficiency.
			computeLogits := promptIndex+nTokens >= len(promptTokens)
			Forward(model, state, tokens, position, computeLogits)

			position += nTokens - 1 // The loop will increment again by 1
			promptIndex += nTokens

			if promptIndex < len(promptTokens) {
				continue // Continue to next batch of prompt tokens
			}
			startGen = time.Now().UnixNano() // Mark end of prompt ingestion

		} else {
			// --- Generation (Single Token) ---

			// Run forward pass for a single token
			Forward(model, state, []int{token}, position, true)
		}

		// --- Sampling and Output ---

		nextToken = sampler(state.Logits.GetData())

		if echo {
			// log inferred token
			fmt.Fprint(os.Stderr, TokenizerReplaceControlCharacters(model.Tokenizer.Decode([]int{nextToken})))
		}

		generatedTokens = append(generatedTokens, nextToken)
		if onTokenGenerated != nil {
			onTokenGenerated(nextToken)
		}

		if _, found := stopTokens[nextToken]; found {
			break
		}

		state.LatestToken = nextToken
		token = nextToken
	}

	elapsedNanos := time.Now().UnixNano() - startNanos
	promptNanos := startGen - startNanos
	genNanos := elapsedNanos
	if startGen > 0 {
		genNanos = elapsedNanos - promptNanos
	}

	promptDuration := time.Duration(promptNanos)
	genDuration := time.Duration(genNanos)

	promptTokensCount := len(promptTokens)
	generatedTokensCount := len(generatedTokens)

	promptTokensPerSecond := float64(promptTokensCount) / promptDuration.Seconds()
	genTokensPerSecond := float64(generatedTokensCount) / genDuration.Seconds()

	fmt.Fprintf(os.Stderr, "\ncontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)\n",
		startPosition+promptIndex+generatedTokensCount, model.Configuration.ContextLength,
		promptTokensPerSecond, promptTokensCount,
		genTokensPerSecond, generatedTokensCount)

	return generatedTokens
}
