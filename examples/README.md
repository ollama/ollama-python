# Running Examples

Run commands from the repository root. Use Python 3.11 or later for the full example set, install [Ollama](https://ollama.com/download), and start its server.

To run against this checkout:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e . rich tqdm 'mcp>=2,<3'
python examples/chat.py
```

Or use uv (including for scripts with inline dependencies):

```sh
uv run --with-editable . examples/chat.py
uv run --with-editable . examples/web-search-mcp.py
```

`--with-editable .` keeps scripts with inline dependencies on the SDK in this checkout. To use a released package instead, install `ollama>=0.6.3` and run the scripts with that environment. System One is available in 0.6.3; System One image inputs are currently on `main` and require this checkout. Python 0.6.3 also accepts model-defined thinking levels; the examples use the GPT-OSS levels `low`, `medium`, and `high`.

Pull the model named in the example before running it, for example `ollama pull gemma4`. Use `ollama list` to see installed models and `ollama show <model>` to check capabilities. Models may require several GB of memory and disk space; pulling is a separate step except in `pull.py`. See the [model library](https://ollama.com/search) for available tags.

| Examples | Model / additional requirements |
| --- | --- |
| Basic chat/generate, logprobs, show, create | `gemma4` |
| Tools, structured text output | `gemma4` |
| Multi-tool, web search | `qwen3` (thinking and tools) |
| GPT-OSS tools, thinking levels | `gpt-oss:20b` |
| GPT-OSS web search | `gpt-oss:120b-cloud`, `ollama signin`, and a web-search API key |
| Thinking chat/generate | `deepseek-r1` |
| Fill in the middle | `codellama:7b-code` |
| Multimodal chat, structured image output | `gemma4` and an image path |
| Multimodal generate | `gemma4`; fetches an XKCD comic over the internet |
| Embeddings | `embeddinggemma`; Ollama v0.11.10 or later |
| Image generation | `x/z-image-turbo`; supported macOS installation; writes `output.png` |
| System One | `nimble`; Ollama v0.35.0 or later |

The `ps.py` and `list.py` examples only inspect models. `create.py` writes a model named `my-assistant`; choose an unused name before running it. `pull.py` downloads `gemma4`. Image generation overwrites `output.png` in the working directory.

Most examples call the local server at `http://localhost:11434`. Set `OLLAMA_HOST` to use another server. A connection error means the server is not reachable; a model-not-found error means you need to pull that exact tag on that server. Structured outputs require a local model; Ollama's cloud does not currently support the `format` schema.

For direct cloud inference, set `OLLAMA_API_KEY` and use `Client(host='https://ollama.com')` with the direct model name, such as `gpt-oss:120b`. Through a signed-in local server, use the `-cloud` tag instead. Web search/fetch always call `https://ollama.com` and require an API key, even when inference is local. Python reads `OLLAMA_API_KEY` automatically. See [cloud authentication](https://docs.ollama.com/cloud/api) and the [API documentation](https://docs.ollama.com/api/introduction).

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
- [multi-tool.py](multi-tool.py) - Using multiple tools, with thinking enabled

#### gpt-oss

- [gpt-oss-tools.py](gpt-oss-tools.py)
- [gpt-oss-tools-stream.py](gpt-oss-tools-stream.py)

### Web search

An API key from Ollama's cloud service is required. You can create one [here](https://ollama.com/settings/keys).

```shell
export OLLAMA_API_KEY="your_api_key_here"
```

- [web-search.py](web-search.py)
- [web-search-gpt-oss.py](web-search-gpt-oss.py) - Using browser research tools with gpt-oss

#### MCP server

The MCP server can be used with an MCP client like Cursor, Cline, Codex, Open WebUI, Goose, and more.

```sh
uv run examples/web-search-mcp.py
```

Configuration to use with an MCP client:

```json
{
  "mcpServers": {
    "web_search": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "path/to/ollama-python/examples/web-search-mcp.py"],
      "env": { "OLLAMA_API_KEY": "your_api_key_here" }
    }
  }
}
```

- [web-search-mcp.py](web-search-mcp.py)

### Multimodal with Images - Chat with a multimodal (image chat) model

- [multimodal-chat.py](multimodal-chat.py)
- [multimodal-generate.py](multimodal-generate.py)

### Image Generation (Experimental) - Generate images with a model

> **Note:** Image generation is experimental and currently only available on macOS.

- [generate-image.py](generate-image.py)

### Structured Outputs - Generate structured outputs with a model

- [structured-outputs.py](structured-outputs.py)
- [async-structured-outputs.py](async-structured-outputs.py)
- [structured-outputs-image.py](structured-outputs-image.py)

### Ollama List - List all downloaded models and their properties

- [list.py](list.py)

### Ollama Show - Display model properties and capabilities

- [show.py](show.py)

### Ollama ps - Show model status with CPU/GPU usage

- [ps.py](ps.py)

### Ollama Pull - Pull a model from Ollama

Requires `tqdm` (installed above).

- [pull.py](pull.py)

### Ollama Create - Create a model from an existing model

- [create.py](create.py)

### Ollama Embed - Generate embeddings with a model

- [embed.py](embed.py)

### Thinking - Enable thinking mode for a model

- [thinking.py](thinking.py)

### Thinking (generate) - Enable thinking mode for a model

- [thinking-generate.py](thinking-generate.py)

### Thinking (levels) - Choose the thinking level

- [thinking-levels.py](thinking-levels.py)

### System One - Score typed questions

- [systemone.py](systemone.py) - Choice, boolean probability, and rubric score questions
