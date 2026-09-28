# Ollama Python Library

The Ollama Python library provides the easiest way to integrate Python 3.8+ projects with [Ollama](https://github.com/ollama/ollama).

## Prerequisites

- [Ollama](https://ollama.com/download) should be installed and running
- Pull a model to use with the library: `ollama pull <model>` e.g. `ollama pull gemma4`
  - See [Ollama.com](https://ollama.com/search) for more information on the models available.

## Install

```sh
pip install ollama
```

## Usage

```python
from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(
  model='gemma4',
  messages=[
    {
      'role': 'user',
      'content': 'Why is the sky blue?',
    },
  ],
)
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)
```

See [_types.py](ollama/_types.py) for more information on the response types.

## Streaming responses

Response streaming can be enabled by setting `stream=True`.

```python
from ollama import chat

stream = chat(
  model='gemma4',
  messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
  stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
```

## Cloud Models

Run larger models by offloading to Ollama’s cloud while keeping your local workflow.

- Supported models: `deepseek-v3.1:671b-cloud`, `gpt-oss:20b-cloud`, `gpt-oss:120b-cloud`, `kimi-k2:1t-cloud`, `qwen3-coder:480b-cloud`, `kimi-k2-thinking` See [Ollama Models - Cloud](https://ollama.com/search?c=cloud) for more information

### Run via local Ollama

1) Sign in (one-time):

```
ollama signin
```

2) Pull a cloud model:

```
ollama pull gpt-oss:120b-cloud
```

3) Make a request:

```python
from ollama import Client

client = Client()

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b-cloud', messages=messages, stream=True):
  print(part.message.content, end='', flush=True)
```

### Cloud API (ollama.com)

Access cloud models directly by pointing the client at `https://ollama.com`.

1) Create an API key from [ollama.com](https://ollama.com/settings/keys) , then set:

```
export OLLAMA_API_KEY=your_api_key
```

2) (Optional) List models available via the API:

```
curl https://ollama.com/api/tags
```

3) Generate a response via the cloud API:

```python
import os
from ollama import Client

client = Client(host='https://ollama.com', headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')})

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
  print(part.message.content, end='', flush=True)
```

## Custom client
A custom client can be created by instantiating `Client` or `AsyncClient` from `ollama`.

All extra keyword arguments are passed into the [`httpx.Client`](https://www.python-httpx.org/api/#client).

```python
from ollama import Client

client = Client(host='http://localhost:11434', headers={'x-some-header': 'some-value'})
response = client.chat(
  model='gemma4',
  messages=[
    {
      'role': 'user',
      'content': 'Why is the sky blue?',
    },
  ],
)
```

## Async client

The `AsyncClient` class is used to make asynchronous requests. It can be configured with the same fields as the `Client` class.

```python
import asyncio
from ollama import AsyncClient


async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  response = await AsyncClient().chat(model='gemma4', messages=[message])


asyncio.run(chat())
```

Setting `stream=True` modifies functions to return a Python asynchronous generator:

```python
import asyncio
from ollama import AsyncClient


async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  async for part in await AsyncClient().chat(model='gemma4', messages=[message], stream=True):
    print(part['message']['content'], end='', flush=True)


asyncio.run(chat())
```

## API

The Ollama Python library's API is designed around the [Ollama REST API](https://github.com/ollama/ollama/blob/main/docs/api.md)

### Chat

```python
ollama.chat(model='gemma4', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
```

### Generate

```python
ollama.generate(model='gemma4', prompt='Why is the sky blue?')
```

### List

```python
ollama.list()
```

### Show

```python
ollama.show('gemma4')
```

### Create

```python
ollama.create(model='example', from_='gemma4', system='You are Mario from Super Mario Bros.')
```

### Copy

```python
ollama.copy('gemma4', 'user/gemma4')
```

### Delete

```python
ollama.delete('gemma4')
```

### Pull

```python
ollama.pull('gemma4')
```

### Push

```python
ollama.push('user/gemma4')
```

### Embed

```python
ollama.embed(model='gemma4', input='The sky is blue because of rayleigh scattering')
```

### Embed (batch)

```python
ollama.embed(model='gemma4', input=['The sky is blue because of rayleigh scattering', 'Grass is green because of chlorophyll'])
```

### Ps

```python
ollama.ps()
```

### System One

```python
import ollama

response = ollama.systemone(
  model='nimble',
  state='Our checkout has returned 500 errors since 9am.',
  questions={
    'team': {
      'type': 'choice',
      'instructions': 'Which team should handle this ticket?',
      'criteria': {'billing': 'Payments and refunds', 'technical': 'Software errors'},
    },
  },
)
print(response.answers['team'])
```

System One requires an Ollama server build that implements `POST /v1/systemone`
and a compatible local model such as `nimble`. It returns one JSON response;
streaming and cloud models are not supported.

`state` and question `instructions` accept text, JSON objects, or arrays. Questions
are evaluated in their supplied order:

- `choice`: 2–26 option keys mapped to descriptions; `None` uses the key as its description.
- `noul`: probability of true, with optional `{"false": "No", "true": "Yes"}` descriptions.
- `score`: 2–26 descriptions ordered lowest to highest; returns a potentially fractional, zero-based score.

Responses contain `model`, `answers`, and `usage.input_tokens` / `usage.output_tokens`.
Confidence measures probability concentration, not calibrated correctness.
Token usage comes from the server; output tokens are not necessarily zero.
Optional `keep_alive` accepts seconds or a duration string. Requests use the client's
existing host, headers, and HTTP error handling. The server validates its body and
model context limits without truncating input.

Available as `ollama.systemone`, `Client.systemone`, and `await AsyncClient.systemone`.
Question and answer types, including `SystemOneResponse`, are exported from `ollama`.
See [the combined question example](examples/systemone.py).

## Errors

Errors are raised if requests return an error status or if an error is detected while streaming.

```python
model = 'does-not-yet-exist'

try:
  ollama.chat(model)
except ollama.ResponseError as e:
  print('Error:', e.error)
  if e.status_code == 404:
    ollama.pull(model)
```
