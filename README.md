# Ollama Python Library

The Ollama Python library provides the easiest way to integrate Python 3.8+ projects with [Ollama](https://github.com/ollama/ollama).

## Install

```sh
pip install ollama
```

## Usage

```python
import ollama
response = ollama.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
```

## Streaming responses

Response streaming can be enabled by setting `stream=True`, modifying function calls to return a Python generator where each part is an object in the stream.

```python
import ollama

stream = ollama.chat(
    model='llama3',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
```

## API

The Ollama Python library's API is designed around the [Ollama REST API](https://github.com/ollama/ollama/blob/main/docs/api.md)

### Chat

```python
ollama.chat(model='llama3', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
```

### Generate

```python
ollama.generate(model='llama3', prompt='Why is the sky blue?')
```

### List

```python
ollama.list()
```

### Show

```python
ollama.show('llama3')
```

### Create

```python
modelfile='''
FROM llama3
SYSTEM You are mario from super mario bros.
'''

ollama.create(model='example', modelfile=modelfile)
```

### Copy

```python
ollama.copy('llama3', 'user/llama3')
```

### Delete

```python
ollama.delete('llama3')
```

### Pull

```python
ollama.pull('llama3')
```

### Push

```python
ollama.push('user/llama3')
```

### Embeddings

```python
ollama.embeddings(model='llama3', prompt='The sky is blue because of rayleigh scattering')
```

### Ps

```python
ollama.ps()
```

## Tool registry

A registry designed to manage a variety of tools, including function(both sync/async), `pydantic.BaseModel`, `typing.TypedDict`, and `typing.NamedTuple`. 

It provides the capability to generate schemas for these tools, which are essential for LLM tool-calling. 

Additionally, it allows for the invocation of tools using their metadata -- name & raw arguments.

- `override`: When set to True, allows the new tool to replace a previously registered tool with the same name.

### Creating a tool registry

```python
import ollama

registry = ollama.ToolRegistry()
```

### Adding tools to the registry instance

```python
import json
import asyncio
from typing import Literal, TypedDict

registry = ollama.ToolRegistry()

@registry.register
def get_flight_times(departure: str, arrival: str) -> str:
  """
  Get flight times.
  :param departure: Departure location code
  :param arrival: Arrival location code
  """
  flights = {
    'NYC-LAX': {'departure': '08:00 AM', 'arrival': '11:30 AM', 'duration': '5h 30m'},
    'LAX-NYC': {'departure': '02:00 PM', 'arrival': '10:30 PM', 'duration': '5h 30m'},
    'LHR-JFK': {'departure': '10:00 AM', 'arrival': '01:00 PM', 'duration': '8h 00m'},
    'JFK-LHR': {'departure': '09:00 PM', 'arrival': '09:00 AM', 'duration': '7h 00m'},
    'CDG-DXB': {'departure': '11:00 AM', 'arrival': '08:00 PM', 'duration': '6h 00m'},
    'DXB-CDG': {'departure': '03:00 AM', 'arrival': '07:30 AM', 'duration': '7h 30m'},
  }

  key = f'{departure}-{arrival}'.upper()
  return json.dumps(flights.get(key, {'error': 'Flight not found'}))

@registry.register
class User(TypedDict): # It can also be `pydantic.BaseModel`, or `typing.NamedTuple`.
    """
    User Information
    :param name: Name of the user
    :param role: Role assigned to the user 
    """
    name: str
    role: Literal['admin', 'developer', 'tester']

"""
# Tools can also be registered using:
registry.register(get_flight_times)
registry.register(User)

# OR

registry.register_multiple(get_flight_times, User)
"""
```

### Get tool schema list

```python
tools = registry.tools
print(json.dumps(tools, indent=3))
```

### Invoking the tool

```python
res = ollama.chat(
    model='llama3-groq-tool-use:latest',
    tools=tools,
    messages=[{
        'role': 'user',
        'content': "What is the flight time from New York (NYC) to Los Angeles (LAX)?"
    }]
)
tool_call = res['message']['tool_calls'][0]['function']
print(f"{tool_call=}")
tool_output = registry.invoke(**tool_call)
print(f"{tool_output=}")
```
```
tool_call={'name': 'get_flight_times', 'arguments': {'arrival': 'LAX', 'departure': 'NYC'}}
tool_output='{"departure": "08:00 AM", "arrival": "11:30 AM", "duration": "5h 30m"}'
```

## Custom client

A custom client can be created with the following fields:

- `host`: The Ollama host to connect to
- `timeout`: The timeout for requests

```python
from ollama import Client
client = Client(host='http://localhost:11434')
response = client.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
```

## Async client

```python
import asyncio
from ollama import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  response = await AsyncClient().chat(model='llama3', messages=[message])

asyncio.run(chat())
```

Setting `stream=True` modifies functions to return a Python asynchronous generator:

```python
import asyncio
from ollama import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  async for part in await AsyncClient().chat(model='llama3', messages=[message], stream=True):
    print(part['message']['content'], end='', flush=True)

asyncio.run(chat())
```

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
