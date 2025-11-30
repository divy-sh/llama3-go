package tokenizer

type Role struct {
	Name string
}

var (
	RoleSystem    = Role{"system"}
	RoleUser      = Role{"user"}
	RoleAssistant = Role{"assistant"}
)

type Message struct {
	Role    Role
	Content string
}
