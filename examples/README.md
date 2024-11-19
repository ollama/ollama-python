# Running Examples

Run the examples in this directory with:
```sh
# Navigate to examples directory
cd examples/

# Run example
python3 <example>.py
```

### Chat - Chat with a model
- [chat.py](chat.py)
- [async-chat.py](async-chat.py)
- [chat-stream.py](chat-stream.py) - Streamed outputs

### Generate - Generate text with a model
- [generate.py](generate.py)
- [async-generate.py](async-generate.py)
- [generate-stream.py](generate-stream.py) - Streamed outputs

### Tools/Function Calling - Call a function with a model
- [tools.py](tools.py) - Simple example of Tools/Function Calling
- [async-tools.py](async-tools.py)

### Multimodal - Chat with a multimodal model
- [multimodal_chat.py](multimodal_chat.py)
- [multimodal_generate.py](multimodal_generate.py)



### Ollama List - List all downloaded models and their properties
- [list.py](list.py)
- [async-list.py](async-list.py)

### Ollama Pull - Pull a model from Ollama
Requirement: `pip install tqdm`

- [pull.py](pull.py) 

### Ollama Create - Create a model from a Modelfile
```python
python create.py <model> <modelfile>
```
- [create.py](create.py) 

See [ollama/docs/modelfile.md](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) for more information on the Modelfile format.

### Fill in the middle
- [fill-in-middle.py](fill-in-middle.py) - Given a prefix and suffix, fill in the middle

### Ollama ps - Show model status with CPU/GPU usage
- [ps.py](ps.py)


