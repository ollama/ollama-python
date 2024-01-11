# Ollama Python Library

The Ollama Python library provides the easiest way to integrate your Python 3 project with [Ollama](https://github.com/jmorganca/ollama).

## Getting Started

Requires Python 3.8 or higher.

```sh
pip install ollama
```

A global default client is provided for convenience and can be used in the same way as the synchronous client.

```python
import ollama
response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
```

```python
import ollama
message = {'role': 'user', 'content': 'Why is the sky blue?'}
for part in ollama.chat(model='llama2', messages=[message], stream=True):
  print(part['message']['content'], end='', flush=True)
```


## Using the Synchronous Client

```python
from ollama import Client
message = {'role': 'user', 'content': 'Why is the sky blue?'}
response = Client().chat(model='llama2', messages=[message])
```

Response streaming can be enabled by setting `stream=True`. This modifies the function to return a Python generator where each part is an object in the stream.

```python
from ollama import Client
message = {'role': 'user', 'content': 'Why is the sky blue?'}
for part in Client().chat(model='llama2', messages=[message], stream=True):
  print(part['message']['content'], end='', flush=True)
```

## Using the Asynchronous Client

```python
import asyncio
from ollama import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  response = await AsyncClient().chat(model='llama2', messages=[message])

asyncio.run(chat())
```

Similar to the synchronous client, setting `stream=True` modifies the function to return a Python asynchronous generator.

```python
import asyncio
from ollama import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  async for part in await AsyncClient().chat(model='llama2', messages=[message], stream=True):
    print(part['message']['content'], end='', flush=True)

asyncio.run(chat())
```

## Handling Errors

Errors are raised if requests return an error status or if an error is detected while streaming.

```python
model = 'does-not-yet-exist'

try:
  ollama.chat(model)
except ollama.ResponseError as e:
  print('Error:', e.content)
  if e.status_code == 404:
    ollama.pull(model)
```
