from ollama._client import AsyncClient, Client
from ollama._utils import create_function_tool
from ollama._tools import (
  ollama_tool, 
  ollama_async_tool, 
  get_ollama_tools, 
  get_ollama_name_async_tools, 
  get_ollama_tools_name,
  get_ollama_tool_description)
from ollama._types import (
  ChatResponse,
  EmbeddingsResponse,
  EmbedResponse,
  GenerateResponse,
  Image,
  ListResponse,
  Message,
  Options,
  ProcessResponse,
  ProgressResponse,
  RequestError,
  ResponseError,
  ShowResponse,
  StatusResponse,
  Tool,
  WebFetchResponse,
  WebSearchResponse,
)

__all__ = [
  'AsyncClient',
  'ChatResponse',
  'Client',
  'EmbedResponse',
  'EmbeddingsResponse',
  'GenerateResponse',
  'Image',
  'ListResponse',
  'Message',
  'Options',
  'ProcessResponse',
  'ProgressResponse',
  'RequestError',
  'ResponseError',
  'ShowResponse',
  'StatusResponse',
  'Tool',
  'WebFetchResponse',
  'WebSearchResponse',
]

_client = Client()

generate = _client.generate
chat = _client.chat
embed = _client.embed
embeddings = _client.embeddings
pull = _client.pull
push = _client.push
create = _client.create
delete = _client.delete
list = _client.list
copy = _client.copy
show = _client.show
ps = _client.ps
web_search = _client.web_search
web_fetch = _client.web_fetch
