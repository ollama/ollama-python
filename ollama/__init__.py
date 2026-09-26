from ollama._client import AsyncClient, Client
from importlib.metadata import version, PackageNotFoundError
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
exists = _client.exists


try:
    __version__ = version("ollama")
except PackageNotFoundError:
    __version__ = "unknown"