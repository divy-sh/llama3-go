package infer

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Categorical sampler
type CategoricalSampler struct {
	rng *rand.Rand
}

func (c CategoricalSampler) SampleToken(logits []float32) int {
	sum := float32(0)
	for _, v := range logits {
		sum += v
	}
	r := c.rng.Float32() * sum
	for i, v := range logits {
		r -= v
		if r <= 0 {
			return i
		}
	}
	return len(logits) - 1
}

// Options struct
type Options struct {
	ModelPath    string
	Prompt       string
	SystemPrompt string
	Interactive  bool
	Temperature  float32
	TopP         float32
	Seed         int64
	MaxTokens    int
	Stream       bool
	Echo         bool
}

// Llama stub
type Llama struct {
	VocabSize int
}

// Sampler selection
func selectSampler(vocabSize int, temperature, topp float32, seed int64) Sampler {
	if temperature == 0 {
		return Sampler.Max()
	}

	rng := rand.New(rand.NewSource(seed))

	if topp <= 0 || topp >= 1 {
		return CategoricalSampler{rng: rng}
	}

	// For simplicity, we treat topp > 0 && topp < 1 same as categorical
	return CategoricalSampler{rng: rng}
}

// Run interactive mode
func runInteractive(model *Llama, sampler Sampler, options Options) {
	state := model.CreateNewState(16)
	reader := bufio.NewReader(os.Stdin)
	conversationTokens := []int{0} // begin-of-text placeholder

	for {
		fmt.Print("> ")
		userText, _ := reader.ReadString('\n')
		userText = strings.TrimSpace(userText)

		switch userText {
		case "/quit", "/exit":
			return
		case "/context":
			fmt.Printf("%d out of %d context tokens used (%d tokens remaining)\n",
				len(conversationTokens),
				options.MaxTokens,
				options.MaxTokens-len(conversationTokens))
			continue
		}

		// Encode user message (stub)
		conversationTokens = append(conversationTokens, encodeMessage(userText)...)
		responseTokens := generateTokens(model, state, conversationTokens, sampler, options)

		for _, token := range responseTokens {
			fmt.Print(model.TokenizerDecode([]int{token}))
		}
		fmt.Println()
		conversationTokens = append(conversationTokens, responseTokens...)
	}
}

// Run instruct mode once
func runInstructOnce(model *Llama, sampler Sampler, options Options) {
	state := model.CreateNewState(16)
	promptTokens := []int{0} // begin-of-text placeholder
	promptTokens = append(promptTokens, encodeMessage(options.Prompt)...)
	responseTokens := generateTokens(model, state, promptTokens, sampler, options)

	for _, token := range responseTokens {
		fmt.Print(model.TokenizerDecode([]int{token}))
	}
	fmt.Println()
}

// Stub for token generation
func generateTokens(model *Llama, state *State, tokens []int, sampler Sampler, options Options) []int {
	// For demonstration, just return some dummy tokens
	return []int{1, 2, 3, 4}
}

// Stub for encoding messages
func encodeMessage(msg string) []int {
	// In reality, this should convert string to token IDs
	return []int{5, 6, 7} // dummy
}

// Parse command-line options
func parseOptions() (Options, error) {
	var opts Options
	flag.StringVar(&opts.ModelPath, "model", "", "Path to model (.gguf)")
	flag.StringVar(&opts.ModelPath, "m", "", "Path to model (.gguf)")
	flag.BoolVar(&opts.Interactive, "interactive", false, "Run in interactive mode")
	flag.BoolVar(&opts.Interactive, "chat", false, "Run in interactive mode")
	flag.StringVar(&opts.Prompt, "prompt", "", "Input prompt")
	flag.StringVar(&opts.Prompt, "p", "", "Input prompt")
	flag.StringVar(&opts.SystemPrompt, "system-prompt", "", "System prompt")
	flag.StringVar(&opts.SystemPrompt, "sp", "", "System prompt")
	flag.Float64Var((*float64)(&opts.Temperature), "temperature", 0.1, "Temperature")
	flag.Float64Var((*float64)(&opts.Temperature), "temp", 0.1, "Temperature")
	flag.Float64Var((*float64)(&opts.TopP), "top-p", 0.95, "Top-p")
	flag.Int64Var(&opts.Seed, "seed", time.Now().UnixNano(), "Random seed")
	flag.IntVar(&opts.MaxTokens, "max-tokens", 512, "Max tokens")
	flag.IntVar(&opts.MaxTokens, "n", 512, "Max tokens")
	flag.BoolVar(&opts.Stream, "stream", true, "Stream tokens")
	flag.BoolVar(&opts.Echo, "echo", false, "Echo tokens")
	flag.Parse()

	if opts.ModelPath == "" {
		return opts, errors.New("missing required --model argument")
	}
	if !opts.Interactive && opts.Prompt == "" {
		return opts, errors.New("missing required --prompt in instruct mode")
	}
	if opts.Temperature < 0 {
		return opts, errors.New("--temperature must be non-negative")
	}
	if opts.TopP < 0 || opts.TopP > 1 {
		return opts, errors.New("--top-p must be in [0,1]")
	}

	opts.ModelPath, _ = filepath.Abs(opts.ModelPath)
	return opts, nil
}

func main() {
	opts, err := parseOptions()
	if err != nil {
		fmt.Println("ERROR:", err)
		os.Exit(1)
	}

	// Load model stub
	model := &Llama{VocabSize: 10000}

	sampler := selectSampler(model.VocabSize, opts.Temperature, opts.TopP, opts.Seed)

	if opts.Interactive {
		runInteractive(model, sampler, opts)
	} else {
		runInstructOnce(model, sampler, opts)
	}
}
