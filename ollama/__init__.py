from ollama._client import Client, AsyncClient
from ollama._types import (
  Options,
  Message,
  Tool,
  GenerateResponse,
  ChatResponse,
  EmbedResponse,
  EmbeddingsResponse,
  StatusResponse,
  ProgressResponse,
  ListResponse,
  ShowResponse,
  ProcessResponse,
  RequestError,
  ResponseError,
)

__all__ = [
  'Client',
  'AsyncClient',
  'Options',
  'Message',
  'Tool',
  'GenerateResponse',
  'ChatResponse',
  'EmbedResponse',
  'EmbeddingsResponse',
  'StatusResponse',
  'ProgressResponse',
  'ListResponse',
  'ShowResponse',
  'ProcessResponse',
  'RequestError',
  'ResponseError',
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
