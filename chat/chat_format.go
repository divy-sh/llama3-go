package chat

import (
	"strings"

	"github.com/divy-sh/llama3-go/tokenizer"
)

// Role defines the role of a message sender.
type Role struct {
	Name string
}

var (
	SYSTEM    = Role{Name: "system"}
	USER      = Role{Name: "user"}
	ASSISTANT = Role{Name: "assistant"}
)

func (r Role) String() string {
	return r.Name
}

type Message struct {
	Role    Role
	Content string
}

// ChatFormat encapsulates the logic for encoding dialogs based on special tokens.
type ChatFormat struct {
	Tokenizer    *tokenizer.Tokenizer
	beginOfText  int
	endHeader    int
	startHeader  int
	endOfTurn    int
	endOfText    int
	endOfMessage int // Only in 3.1
	StopTokens   map[int]struct{}
}

// NewChatFormat creates a new ChatFormat instance.
func NewChatFormat(t *tokenizer.Tokenizer) *ChatFormat {
	specialTokens := t.GetSpecialTokens()

	cf := &ChatFormat{
		Tokenizer:  t,
		StopTokens: make(map[int]struct{}),
	}

	cf.beginOfText = specialTokens["<|begin_of_text|>"]
	cf.startHeader = specialTokens["<|start_header_id|>"]
	cf.endHeader = specialTokens["<|end_header_id|>"]
	cf.endOfTurn = specialTokens["<|eot_id|>"]
	cf.endOfText = specialTokens["<|end_of_text|>"]

	if val, ok := specialTokens["<|eom_id|>"]; ok {
		cf.endOfMessage = val
	} else {
		cf.endOfMessage = -1
	}

	cf.StopTokens[cf.endOfText] = struct{}{}
	cf.StopTokens[cf.endOfTurn] = struct{}{}

	return cf
}

// GetTokenizer returns the underlying tokenizer.
func (cf *ChatFormat) GetTokenizer() *tokenizer.Tokenizer {
	return cf.Tokenizer
}

// GetStopTokens returns the set of tokens that should halt generation.
func (cf *ChatFormat) GetStopTokens() map[int]struct{} {
	return cf.StopTokens
}

// EncodeHeader encodes the role header, e.g., <|start_header_id|>user<|end_header_id|>\n
func (cf *ChatFormat) EncodeHeader(message Message) []int {
	tokens := []int{cf.startHeader}
	tokens = append(tokens, cf.Tokenizer.EncodeAsList(message.Role.Name)...)
	tokens = append(tokens, cf.endHeader)
	tokens = append(tokens, cf.Tokenizer.EncodeAsList("\n")...)
	return tokens
}

// EncodeMessage encodes a full message with header and EOT.
func (cf *ChatFormat) EncodeMessage(message Message) []int {
	tokens := cf.EncodeHeader(message)
	tokens = append(tokens, cf.Tokenizer.EncodeAsList(strings.TrimSpace(message.Content))...)
	tokens = append(tokens, cf.endOfTurn)
	return tokens
}

// EncodeDialogPrompt encodes a full dialog into the model's required prompt format.
func (cf *ChatFormat) EncodeDialogPrompt(appendAssistantTurn bool, dialog []Message) []int {
	tokens := []int{cf.beginOfText}
	for _, message := range dialog {
		tokens = append(tokens, cf.EncodeMessage(message)...)
	}
	if appendAssistantTurn {
		// Add the start of an assistant message for the model to complete.
		tokens = append(tokens, cf.EncodeHeader(Message{Role: ASSISTANT, Content: ""})...)
	}
	return tokens
}
