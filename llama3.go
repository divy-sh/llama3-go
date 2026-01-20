package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/divy-sh/llama3-go/aot"
	"github.com/divy-sh/llama3-go/chat"
	"github.com/divy-sh/llama3-go/llama"
	"github.com/divy-sh/llama3-go/model"
	"github.com/divy-sh/llama3-go/tensor"
	"github.com/divy-sh/llama3-go/util"
)

// BATCH_SIZE used in prompt evaluation.
var BATCH_SIZE = 16

func init() {

	if bsStr := os.Getenv("LLAMA_BATCHSIZE"); bsStr != "" {
		if bs, err := strconv.Atoi(bsStr); err == nil {
			BATCH_SIZE = bs
		}
	}
}

func selectSampler(vocabularySize int, temperature float32, topp float32, rngSeed int64) model.Sampler {
	if temperature == 0.0 {
		// Greedy argmax sampling
		return func(logits tensor.FloatTensor) int {
			return model.ARGMAX.SampleToken(logits)
		}
	}

	// Sampler implementation (assuming internal implementation details for CategoricalSampler, ToppSampler)
	rng := util.NewRandomGenerator(rngSeed) // Assuming a utility for random generation

	var innerSampler model.Sampler
	if topp <= 0 || topp >= 1 {
		innerSampler = &model.CategoricalSampler{Rng: rng}
	} else {
		innerSampler = model.NewToppSampler(vocabularySize, topp, rng)
	}

	return func(logits []float32) int {
		// Apply the temperature to the logits
		llama.DivideInPlace(logits, temperature) // Assuming a helper function
		// Apply softmax to the logits to get the probabilities
		llama.SoftmaxInPlace(logits) // Assuming a helper function
		return innerSampler.SampleToken(logits)
	}
}

// runInteractive implements the chat mode.
func runInteractive(model *llama.Llama, sampler Sampler, options Options) {
	var state *llama.State
	conversationTokens := []int{}
	chatFormat := chat.NewChatFormat(model.Tokenizer) // Assuming NewChatFormat exists
	conversationTokens = append(conversationTokens, chatFormat.BeginOfTest)

	if options.SystemPrompt != "" {
		conversationTokens = append(conversationTokens, chatFormat.EncodeMessage(chat.NewMessage(chat.SYSTEM, options.SystemPrompt))...)
	}

	startPosition := 0
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("Entering interactive chat mode. Type '/quit' or '/exit' to leave.")

loop:
	for {
		fmt.Print("> ")
		os.Stdout.Flush()

		userText, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			log.Printf("Error reading input: %v", err)
			break loop
		}
		userText = strings.TrimSpace(userText)

		switch userText {
		case "/quit", "/exit":
			break loop
		case "/context":
			fmt.Printf("%d out of %d context tokens used (%d tokens remaining)\n",
				len(conversationTokens),
				options.MaxTokens,
				options.MaxTokens-len(conversationTokens))
			continue
		}

		if state == nil {
			state = model.CreateNewState(BATCH_SIZE)
		}

		conversationTokens = append(conversationTokens, chatFormat.EncodeMessage(llama.NewMessage(chat.USER, userText))...)
		conversationTokens = append(conversationTokens, chatFormat.EncodeHeader(llama.NewMessage(chat.ASSISTANT, ""))...)

		stopTokens := chatFormat.GetStopTokens()

		// Get a slice of the prompt tokens to process this turn
		currentPrompt := conversationTokens[startPosition:]

		responseTokens := llama.GenerateTokens(model, state, startPosition, currentPrompt, stopTokens, options.MaxTokens, sampler, options.Echo, func(token int) {
			if options.Stream {
				if !model.Tokenizer.IsSpecialToken(token) {
					fmt.Print(model.Tokenizer.Decode([]int{token}))
					os.Stdout.Flush()
				}
			}
		})

		// Include stop token in the prompt history.
		conversationTokens = append(conversationTokens, responseTokens...)
		startPosition = len(conversationTokens)

		var stopToken *int
		if len(responseTokens) > 0 && slices.Contains(stopTokens, responseTokens[len(responseTokens)-1]) {
			// Found stop token
			val := responseTokens[len(responseTokens)-1]
			stopToken = &val
			responseTokens = responseTokens[:len(responseTokens)-1] // Remove from displayed response
		}

		if !options.Stream {
			responseText := model.Tokenizer.Decode(responseTokens)
			fmt.Println(responseText)
		}
		fmt.Println() // Newline after response for clean prompt display

		if stopToken == nil {
			fmt.Fprintln(os.Stderr, "Ran out of context length...")
			break
		}
	}
}

// runInstructOnce implements the single-turn instruct mode.
func runInstructOnce(model *llama.Llama, sampler Sampler, options Options) {
	state := model.CreateNewState(BATCH_SIZE)
	chatFormat := llama.NewChatFormat(model.Tokenizer)

	promptTokens := []int{}
	promptTokens = append(promptTokens, chatFormat.BeginOfTest)

	if options.SystemPrompt != "" {
		promptTokens = append(promptTokens, chatFormat.EncodeMessage(llama.NewMessage(chat.SYSTEM, options.SystemPrompt))...)
	}
	promptTokens = append(promptTokens, chatFormat.EncodeMessage(llama.NewMessage(chat.USER, options.Prompt))...)
	promptTokens = append(promptTokens, chatFormat.EncodeHeader(llama.NewMessage(chat.ASSISTANT, ""))...)

	stopTokens := chatFormat.GetStopTokens()

	responseTokens := llama.GenerateTokens(model, state, 0, promptTokens, stopTokens, options.MaxTokens, sampler, options.Echo, func(token int) {
		if options.Stream {
			if !model.Tokenizer.IsSpecialToken(token) {
				fmt.Print(model.Tokenizer.Decode([]int{token}))
				os.Stdout.Flush()
			}
		}
	})

	if len(responseTokens) > 0 && slices.Contains(stopTokens, responseTokens[len(responseTokens)-1]) {
		responseTokens = responseTokens[:len(responseTokens)-1]
	}

	if !options.Stream {
		responseText := model.Tokenizer.Decode(responseTokens)
		fmt.Println(responseText)
	}
	fmt.Println() // Ensure a final newline
}

// Options struct holds all runtime configuration.
type Options struct {
	ModelPath    string
	Prompt       string
	SystemPrompt string
	Interactive  bool
	Temperature  float32
	Topp         float32
	Seed         int64
	MaxTokens    int
	Stream       bool
	Echo         bool
}

const defaultMaxTokens = 512

func NewOptions(modelPath, prompt, systemPrompt string, interactive bool, temperature, topp float32, seed int64, maxTokens int, stream, echo bool) Options {
	options := Options{
		ModelPath:    modelPath,
		Prompt:       prompt,
		SystemPrompt: systemPrompt,
		Interactive:  interactive,
		Temperature:  temperature,
		Topp:         topp,
		Seed:         seed,
		MaxTokens:    maxTokens,
		Stream:       stream,
		Echo:         echo,
	}
	options.require(modelPath != "", "Missing argument: --model <path> is required")
	options.require(interactive || prompt != "", "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"")
	options.require(0 <= temperature, "Invalid argument: --temperature must be non-negative")
	options.require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]")
	return options
}

func (o Options) require(condition bool, format string, args ...interface{}) {
	if !condition {
		fmt.Fprintf(os.Stderr, "ERROR "+format+"\n", args...)
		fmt.Fprintln(os.Stderr)
		printUsage(os.Stderr)
		os.Exit(1)
	}
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, "Usage: go run Llama3.go [options]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Options:")
	fmt.Fprintln(w, "  --model, -m <path>            required, path to .gguf file")
	fmt.Fprintln(w, "  --interactive, --chat, -i     run in chat mode")
	fmt.Fprintln(w, "  --instruct                    run in instruct (once) mode, default mode")
	fmt.Fprintln(w, "  --prompt, -p <string>         input prompt")
	fmt.Fprintln(w, "  --system-prompt, -sp <string> (optional) system prompt")
	fmt.Fprintln(w, "  --temperature, -temp <float>  temperature in [0,inf], default 0.1")
	fmt.Fprintln(w, "  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95")
	fmt.Fprintln(w, "  --seed <long>                 random seed, default current nano time")
	fmt.Fprintln(w, "  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default", defaultMaxTokens)
	fmt.Fprintln(w, "  --stream <boolean>            print tokens during generation; may cause encoding artifacts for non ASCII text, default true")
	fmt.Fprintln(w, "  --echo <boolean>              print ALL tokens to stderr, if true, recommended to set --stream=false, default false")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Examples:")
	fmt.Fprintln(w, "  go run Llama3.go --model llama3.2-1b-q4_0.gguf --prompt \"Tell me a joke\"")
	// ... (rest of examples)
}

func parseOptions(args []string) Options {
	var prompt string
	var systemPrompt string
	temperature := float32(0.1)
	topp := float32(0.95)
	var modelPath string
	seed := time.Now().UnixNano()
	maxTokens := defaultMaxTokens
	interactive := false
	stream := true
	echo := false

	for i := 0; i < len(args); i++ {
		optionName := args[i]
		if !strings.HasPrefix(optionName, "-") {
			NewOptions("", "", "", false, 0, 0, 0, 0, false, false).require(false, "Invalid option %s", optionName)
		}

		switch optionName {
		case "--interactive", "--chat", "-i":
			interactive = true
		case "--instruct":
			interactive = false
		case "--help", "-h":
			printUsage(os.Stdout)
			os.Exit(0)
		default:
			var nextArg string
			if parts := strings.SplitN(optionName, "=", 2); len(parts) == 2 {
				optionName = parts[0]
				nextArg = parts[1]
			} else {
				NewOptions("", "", "", false, 0, 0, 0, 0, false, false).require(i+1 < len(args), "Missing argument for option %s", optionName)
				nextArg = args[i+1]
				i++ // skip arg
			}

			switch optionName {
			case "--prompt", "-p":
				prompt = nextArg
			case "--system-prompt", "-sp":
				systemPrompt = nextArg
			case "--temperature", "--temp":
				if val, err := strconv.ParseFloat(nextArg, 32); err == nil {
					temperature = float32(val)
				}
			case "--top-p":
				if val, err := strconv.ParseFloat(nextArg, 32); err == nil {
					topp = float32(val)
				}
			case "--model", "-m":
				modelPath = nextArg
			case "--seed", "-s":
				if val, err := strconv.ParseInt(nextArg, 10, 64); err == nil {
					seed = val
				}
			case "--max-tokens", "-n":
				if val, err := strconv.Atoi(nextArg); err == nil {
					maxTokens = val
				}
			case "--stream":
				stream = strings.EqualFold(nextArg, "true") // Standard Go boolean parsing
			case "--echo":
				echo = strings.EqualFold(nextArg, "true")
			default:
				NewOptions("", "", "", false, 0, 0, 0, 0, false, false).require(false, "Unknown option: %s", optionName)
			}
		}
	}
	return NewOptions(modelPath, prompt, systemPrompt, interactive, temperature, topp, seed, maxTokens, stream, echo)
}

func run() {
	options := parseOptions(os.Args[1:])

	modelPath := options.ModelPath
	maxTokens := options.MaxTokens

	// Use filepath.Abs for canonical path
	absModelPath, err := filepath.Abs(modelPath)
	if err != nil {
		log.Fatalf("Error resolving model path: %v", err)
	}

	// Try using AOT (Ahead-of-Time compilation/preloading), assuming it's implemented.
	model, _ := aot.TryUsePreLoaded(absModelPath, maxTokens)
	if model == nil {
		// Fallback to fully parse and load the specified file.
		model, err = llama.LoadModel(absModelPath, maxTokens, true) // Assuming LoadModel exists
		if err != nil {
			log.Fatalf("Failed to load model %s: %v", absModelPath, err)
		}
	}

	sampler := selectSampler(model.Configuration.VocabularySize, options.Temperature, options.Topp, options.Seed)

	if options.Interactive {
		runInteractive(model, sampler, options)
	} else {
		runInstructOnce(model, sampler, options)
	}
}
