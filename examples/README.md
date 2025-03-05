# Running Examples

Run the examples in this directory with:
```sh
# Run example
python3 examples/<example>.py
```

See [ollama/docs/api.md](https://github.com/ollama/ollama/blob/main/docs/api.md) for full API documentation

### Chat - Chat with a model
- [chat.py](chat.py)
- [async-chat.py](async-chat.py)
- [chat-stream.py](chat-stream.py) - Streamed outputs
- [chat-with-history.py](chat-with-history.py) - Chat with model and maintain history of the conversation


### Generate - Generate text with a model
- [generate.py](generate.py)
- [async-generate.py](async-generate.py)
- [generate-stream.py](generate-stream.py) - Streamed outputs
- [fill-in-middle.py](fill-in-middle.py) - Given a prefix and suffix, fill in the middle


### Tools/Function Calling - Call a function with a model
- [tools.py](tools.py) - Simple example of Tools/Function Calling
- [async-tools.py](async-tools.py)


### Multimodal with Images - Chat with a multimodal (image chat) model
- [multimodal-chat.py](multimodal-chat.py)
- [multimodal-generate.py](multimodal-generate.py)


### Structured Outputs - Generate structured outputs with a model
- [structured-outputs.py](structured-outputs.py)
- [async-structured-outputs.py](async-structured-outputs.py)
- [structured-outputs-image.py](structured-outputs-image.py)


### Ollama List - List all downloaded models and their properties
- [list.py](list.py)


### Ollama ps - Show model status with CPU/GPU usage
- [ps.py](ps.py)


### Ollama Pull - Pull a model from Ollama
Requirement: `pip install tqdm`
- [pull.py](pull.py) 


### Ollama Create - Create a model from a Modelfile
- [create.py](create.py) 



### Ollama Embed - Generate embeddings with a model
- [embed.py](embed.py)

