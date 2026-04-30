from importlib.metadata import version as _pkg_version
from ollama._client import AsyncClient, Client
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

__version__ = _pkg_version('ollama')

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
    '__version__',
    'version',
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
show = _client.show
copy = _client.copy
ps = _client.ps
version = _client.version   # Expose the version method for users to check the Ollama version
