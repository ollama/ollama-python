# Running Examples

Run the examples in this directory with:
```sh
# Navigate to examples directory
cd examples/

# Run example
python3 <example>.py
```

### Chat
- [chat.py](chat.py) - Basic chat with model
- [chat-stream.py](chat-stream.py) - Stream chat with model
- [async-chat.py](async-chat.py) - Async chat with model

### Generate
- [generate.py](generate.py) - Generate text with model
- [generate-stream.py](generate-stream.py) - Stream generate text with model
- [async-generate.py](async-generate.py) - Async generate text with model

### List
- [list.py](list.py) - List all downloaded models and their properties
- [async-list.py](async-list.py) - Async list all downloaded models and their properties

### Fill in the middle
- [fill-in-middle.py](fill-in-middle.py) - Fill in the middle with model


### Multimodal
- [multimodal.py](multimodal.py) - Multimodal chat with model

### Pull Progress
Requirement: `pip install tqdm`

- [pull-progress.py](pull-progress.py) - Pull progress with model

### Ollama create (create a model)
- [create.py](create.py) - Create a model

### Ollama ps (show model status - cpu/gpu usage)
- [ollama-ps.py](ollama-ps.py) - Ollama ps

### Tools/Function Calling
- [tools.py](tools.py) - Simple example of Tools/Function Calling
- [async-tools.py](async-tools.py) - Async example of Tools/Function Calling

## Configuring Clients
Custom parameters can be passed to the client when initializing:
```python
import ollama
client = ollama.Client(
  host='http://localhost:11434',
  timeout=10.0, # Default: None
  follow_redirects=True, # Default: True
  headers={'x-some-header': 'some-value'}
)
```

Similarly, the `AsyncClient` class can be configured with the same parameters.

