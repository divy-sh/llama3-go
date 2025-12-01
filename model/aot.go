package model

type PartialModel struct {
	fileName         string
	model            Llama
	tensorDataOffset int64
	tensorInfos      map[string]GGUF.GGUFTensorInfo
}

// TODO : replace the preloaded model path from config
var (
	PRELOADED_GGUF = preLoadGGUF("model/gguf/llama3")
)

func preLoadGGUF(path string) GGUF.GGUF, error {
	if path == "" {
		return nil
	}
	gguf, err := GGUF.loadModel(path)
	if err != nil {
		return nil, err
	}
	return PartialModel{
		fileName:         path,
		model:            ModelLoader.LoadModel(path),
		tensorDataOffset: gguf.GetTensorDataOffset(),
		tensorInfos:      gguf.GetTensorInfos(),
	}, nil
}

func UsePreloaded() Llama {
	partialModel := PRELOADED_GGUF.(PartialModel)
	baseModel := preLoaded.model()
	
}