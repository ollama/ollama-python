from importlib import metadata as _metadata

try:
  __version__: str = _metadata.version('ollama')
except _metadata.PackageNotFoundError:
  __version__ = '0.0.0'

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

__all__ = [
  '__version__',
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
  'version',
  'WebFetchResponse',
  'WebSearchResponse',
]

_client = Client()
_async_client = AsyncClient()

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
exists = _client.exists
web_search = _client.web_search
web_fetch = _client.web_fetch


def version() -> str:
  """Return the running Ollama server version string.

  Hits the ``/api/version`` endpoint on the local Ollama server and returns
  the version string (e.g. ``"0.18.2"``).

  This is distinct from :data:`__version__`, which is the version of the
  *Python client package* itself.

  Returns:
    The Ollama server version string.

  Example::

    import ollama

    print(ollama.__version__)   # "0.6.1"  — client package version
    print(ollama.version())     # "0.18.2" — running server version
  """
  return _client.version()


async def async_version() -> str:
  """Async variant of :func:`version`.

  Returns:
    The Ollama server version string.
  """
  return await _async_client.version()
