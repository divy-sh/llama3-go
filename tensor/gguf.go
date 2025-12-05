package tensor

type GGUFTensorInfo struct {
	Name       string
	Dimensions []int
	GgmlType   GGMLType
	offset     int64
}

var (
	ggufMagic             = 0x46554747
	defaultAlignment      = 32
	supportedGGUFVersions = [2]int{1, 0}
)

type GGUF struct {
	magic            int
	version          int
	tensorCount      int
	alignment        int
	metadataKVCount  int
	metadata         map[string]interface{}
	tensorInfo       map[string]GGUFTensorInfo
	tensorDataOffset int64
}

func (g *GGUF) GetTensorInfo() map[string]GGUFTensorInfo {
	return g.tensorInfo
}

func (g *GGUF) GetTensorDataOffset() int64 {
	return g.tensorDataOffset
}

func (g *GGUF) GetMetadata() map[string]interface{} {
	return g.metadata
}
