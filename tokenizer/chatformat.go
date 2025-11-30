package tokenizer

import "strings"

type ChatFormat struct {
	beginOfText  int
	endOfText    int
	startHeader  int
	endHeader    int
	endOfTurn    int
	endOfMessage int
	stopTokens   map[int]bool
	tokenizer    Tokenizer
}

func NewChatFormat(tokenizer Tokenizer) *ChatFormat {
	specialTokens := tokenizer.GetSpecialTokens()

	beginOfText := specialTokens["<|begin_of_text|>"]
	endOfText := specialTokens["<|end_of_text|>"]
	startHeader := specialTokens["<|start_header|>"]
	endHeader := specialTokens["<|end_header|>"]
	endOfTurn := specialTokens["<|end_of_turn|>"]
	endOfMessage := specialTokens["<|end_of_message|>"]

	stopTokens := map[int]bool{
		endOfText:    true,
		endOfMessage: true,
	}
	return &ChatFormat{
		beginOfText:  beginOfText,
		endOfText:    endOfText,
		startHeader:  startHeader,
		endHeader:    endHeader,
		endOfTurn:    endOfTurn,
		endOfMessage: endOfMessage,
		stopTokens:   stopTokens,
		tokenizer:    tokenizer,
	}
}

func (cf *ChatFormat) GetStopTokens() *map[int]bool {
	return &cf.stopTokens
}

func (cf *ChatFormat) GetTokenizer() *Tokenizer {
	return &cf.tokenizer
}

func (cf *ChatFormat) EncodeHeader(msg Message) []int {
	tokens := []int{cf.startHeader}

	tokens = append(tokens, cf.tokenizer.EncodeAsList(msg.Role.Name)...)
	tokens = append(tokens, cf.endHeader)

	tokens = append(tokens, cf.tokenizer.EncodeAsList("\n")...)
	return tokens
}

func (cf *ChatFormat) EncodeMessage(msg Message) *[]int {
	tokens := cf.EncodeHeader(msg)
	tokens = append(tokens, cf.tokenizer.EncodeAsList(strings.TrimSpace(msg.Content))...)
	tokens = append(tokens, cf.endOfTurn)
	return &tokens
}

func (cf *ChatFormat) EncodeDialogPrompt(appendAssistantTurn bool, dialog []Message) *[]int {
	tokens := []int{cf.beginOfText}

	for _, m := range dialog {
		tokens = append(tokens, *cf.EncodeMessage(m)...)
	}

	if appendAssistantTurn {
		empty := Message{RoleAssistant, ""}
		tokens = append(tokens, cf.EncodeHeader(empty)...)
	}

	return &tokens
}
