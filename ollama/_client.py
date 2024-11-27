import os
import io
import json
import platform
import ipaddress
import urllib.parse
from os import PathLike
from pathlib import Path
from hashlib import sha256

from typing import (
  Any,
  Callable,
  Literal,
  Mapping,
  Optional,
  Sequence,
  Type,
  TypeVar,
  Union,
  overload,
)

import sys


from ollama._utils import convert_function_to_tool

if sys.version_info < (3, 9):
  from typing import Iterator, AsyncIterator
else:
  from collections.abc import Iterator, AsyncIterator

from importlib import metadata

try:
  __version__ = metadata.version('ollama')
except metadata.PackageNotFoundError:
  __version__ = '0.0.0'

import httpx

from ollama._types import (
  ChatRequest,
  ChatResponse,
  CreateRequest,
  CopyRequest,
  DeleteRequest,
  EmbedRequest,
  EmbedResponse,
  EmbeddingsRequest,
  EmbeddingsResponse,
  GenerateRequest,
  GenerateResponse,
  Image,
  ListResponse,
  Message,
  Options,
  ProcessResponse,
  ProgressResponse,
  PullRequest,
  PushRequest,
  RequestError,
  ResponseError,
  ShowRequest,
  ShowResponse,
  StatusResponse,
  Tool,
)


T = TypeVar('T')


class BaseClient:
  def __init__(
    self,
    client,
    host: Optional[str] = None,
    follow_redirects: bool = True,
    timeout: Any = None,
    headers: Optional[Mapping[str, str]] = None,
    **kwargs,
  ) -> None:
    """
    Creates a httpx client. Default parameters are the same as those defined in httpx
    except for the following:
    - `follow_redirects`: True
    - `timeout`: None
    `kwargs` are passed to the httpx client.
    """

    self._client = client(
      base_url=_parse_host(host or os.getenv('OLLAMA_HOST')),
      follow_redirects=follow_redirects,
      timeout=timeout,
      # Lowercase all headers to ensure override
      headers={
        k.lower(): v
        for k, v in {
          **(headers or {}),
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'User-Agent': f'ollama-python/{__version__} ({platform.machine()} {platform.system().lower()}) Python/{platform.python_version()}',
        }.items()
      },
      **kwargs,
    )


class Client(BaseClient):
  def __init__(self, host: Optional[str] = None, **kwargs) -> None:
    super().__init__(httpx.Client, host, **kwargs)

  def _request_raw(self, *args, **kwargs):
    r = self._client.request(*args, **kwargs)
    try:
      r.raise_for_status()
    except httpx.HTTPStatusError as e:
      raise ResponseError(e.response.text, e.response.status_code) from None
    return r

  @overload
  def _request(
    self,
    cls: Type[T],
    *args,
    stream: Literal[False] = False,
    **kwargs,
  ) -> T: ...

  @overload
  def _request(
    self,
    cls: Type[T],
    *args,
    stream: Literal[True] = True,
    **kwargs,
  ) -> Iterator[T]: ...

  @overload
  def _request(
    self,
    cls: Type[T],
    *args,
    stream: bool = False,
    **kwargs,
  ) -> Union[T, Iterator[T]]: ...

  def _request(
    self,
    cls: Type[T],
    *args,
    stream: bool = False,
    **kwargs,
  ) -> Union[T, Iterator[T]]:
    if stream:

      def inner():
        with self._client.stream(*args, **kwargs) as r:
          try:
            r.raise_for_status()
          except httpx.HTTPStatusError as e:
            e.response.read()
            raise ResponseError(e.response.text, e.response.status_code) from None

          for line in r.iter_lines():
            part = json.loads(line)
            if err := part.get('error'):
              raise ResponseError(err)
            yield cls(**part)

      return inner()

    return cls(**self._request_raw(*args, **kwargs).json())

  @overload
  def generate(
    self,
    model: str = '',
    prompt: str = '',
    suffix: str = '',
    *,
    system: str = '',
    template: str = '',
    context: Optional[Sequence[int]] = None,
    stream: Literal[False] = False,
    raw: bool = False,
    format: Optional[Literal['', 'json']] = None,
    images: Optional[Sequence[Union[str, bytes]]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> GenerateResponse: ...

  @overload
  def generate(
    self,
    model: str = '',
    prompt: str = '',
    suffix: str = '',
    *,
    system: str = '',
    template: str = '',
    context: Optional[Sequence[int]] = None,
    stream: Literal[True] = True,
    raw: bool = False,
    format: Optional[Literal['', 'json']] = None,
    images: Optional[Sequence[Union[str, bytes]]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Iterator[GenerateResponse]: ...

  def generate(
    self,
    model: str = '',
    prompt: Optional[str] = None,
    suffix: Optional[str] = None,
    *,
    system: Optional[str] = None,
    template: Optional[str] = None,
    context: Optional[Sequence[int]] = None,
    stream: bool = False,
    raw: Optional[bool] = None,
    format: Optional[Literal['', 'json']] = None,
    images: Optional[Sequence[Union[str, bytes]]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
    """
    Create a response using the requested model.

    Raises `RequestError` if a model is not provided.

    Raises `ResponseError` if the request could not be fulfilled.

    Returns `GenerateResponse` if `stream` is `False`, otherwise returns a `GenerateResponse` generator.
    """

    return self._request(
      GenerateResponse,
      'POST',
      '/api/generate',
      json=GenerateRequest(
        model=model,
        prompt=prompt,
        suffix=suffix,
        system=system,
        template=template,
        context=context,
        stream=stream,
        raw=raw,
        format=format,
        images=[Image(value=image) for image in images] if images else None,
        options=options,
        keep_alive=keep_alive,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  @overload
  def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
    stream: Literal[False] = False,
    format: Optional[Literal['', 'json']] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> ChatResponse: ...

  @overload
  def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
    stream: Literal[True] = True,
    format: Optional[Literal['', 'json']] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Iterator[ChatResponse]: ...

  def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
    stream: bool = False,
    format: Optional[Literal['', 'json']] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Union[ChatResponse, Iterator[ChatResponse]]:
    """
    Create a chat response using the requested model.

    Args:
      tools:
        A JSON schema as a dict, an Ollama Tool or a Python Function.
        Python functions need to follow Google style docstrings to be converted to an Ollama Tool.
        For more information, see: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
      stream: Whether to stream the response.
      format: The format of the response.

    Example:
      def add_two_numbers(a: int, b: int) -> int:
        '''
        Add two numbers together.

        Args:
          a: First number to add
          b: Second number to add

        Returns:
          int: The sum of a and b
        '''
        return a + b

      client.chat(model='llama3.2', tools=[add_two_numbers], messages=[...])

    Raises `RequestError` if a model is not provided.

    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ChatResponse` if `stream` is `False`, otherwise returns a `ChatResponse` generator.
    """

    return self._request(
      ChatResponse,
      'POST',
      '/api/chat',
      json=ChatRequest(
        model=model,
        messages=[message for message in _copy_messages(messages)],
        tools=[tool for tool in _copy_tools(tools)],
        stream=stream,
        format=format,
        options=options,
        keep_alive=keep_alive,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  def embed(
    self,
    model: str = '',
    input: Union[str, Sequence[str]] = '',
    truncate: Optional[bool] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> EmbedResponse:
    return self._request(
      EmbedResponse,
      'POST',
      '/api/embed',
      json=EmbedRequest(
        model=model,
        input=input,
        truncate=truncate,
        options=options,
        keep_alive=keep_alive,
      ).model_dump(exclude_none=True),
    )

  def embeddings(
    self,
    model: str = '',
    prompt: Optional[str] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> EmbeddingsResponse:
    """
    Deprecated in favor of `embed`.
    """
    return self._request(
      EmbeddingsResponse,
      'POST',
      '/api/embeddings',
      json=EmbeddingsRequest(
        model=model,
        prompt=prompt,
        options=options,
        keep_alive=keep_alive,
      ).model_dump(exclude_none=True),
    )

  @overload
  def pull(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[False] = False,
  ) -> ProgressResponse: ...

  @overload
  def pull(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[True] = True,
  ) -> Iterator[ProgressResponse]: ...

  def pull(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: bool = False,
  ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    return self._request(
      ProgressResponse,
      'POST',
      '/api/pull',
      json=PullRequest(
        model=model,
        insecure=insecure,
        stream=stream,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  @overload
  def push(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[False] = False,
  ) -> ProgressResponse: ...

  @overload
  def push(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[True] = True,
  ) -> Iterator[ProgressResponse]: ...

  def push(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: bool = False,
  ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    return self._request(
      ProgressResponse,
      'POST',
      '/api/push',
      json=PushRequest(
        model=model,
        insecure=insecure,
        stream=stream,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  @overload
  def create(
    self,
    model: str,
    path: Optional[Union[str, PathLike]] = None,
    modelfile: Optional[str] = None,
    *,
    quantize: Optional[str] = None,
    stream: Literal[False] = False,
  ) -> ProgressResponse: ...

  @overload
  def create(
    self,
    model: str,
    path: Optional[Union[str, PathLike]] = None,
    modelfile: Optional[str] = None,
    *,
    quantize: Optional[str] = None,
    stream: Literal[True] = True,
  ) -> Iterator[ProgressResponse]: ...

  def create(
    self,
    model: str,
    path: Optional[Union[str, PathLike]] = None,
    modelfile: Optional[str] = None,
    *,
    quantize: Optional[str] = None,
    stream: bool = False,
  ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    if (realpath := _as_path(path)) and realpath.exists():
      modelfile = self._parse_modelfile(realpath.read_text(), base=realpath.parent)
    elif modelfile:
      modelfile = self._parse_modelfile(modelfile)
    else:
      raise RequestError('must provide either path or modelfile')

    return self._request(
      ProgressResponse,
      'POST',
      '/api/create',
      json=CreateRequest(
        model=model,
        modelfile=modelfile,
        stream=stream,
        quantize=quantize,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  def _parse_modelfile(self, modelfile: str, base: Optional[Path] = None) -> str:
    base = Path.cwd() if base is None else base

    out = io.StringIO()
    for line in io.StringIO(modelfile):
      command, _, args = line.partition(' ')
      if command.upper() not in ['FROM', 'ADAPTER']:
        print(line, end='', file=out)
        continue

      path = Path(args.strip()).expanduser()
      path = path if path.is_absolute() else base / path
      if path.exists():
        args = f'@{self._create_blob(path)}\n'
      print(command, args, end='', file=out)

    return out.getvalue()

  def _create_blob(self, path: Union[str, Path]) -> str:
    sha256sum = sha256()
    with open(path, 'rb') as r:
      while True:
        chunk = r.read(32 * 1024)
        if not chunk:
          break
        sha256sum.update(chunk)

    digest = f'sha256:{sha256sum.hexdigest()}'

    with open(path, 'rb') as r:
      self._request_raw('POST', f'/api/blobs/sha256:{digest}', content=r)

    return digest

  def list(self) -> ListResponse:
    return self._request(
      ListResponse,
      'GET',
      '/api/tags',
    )

  def delete(self, model: str) -> StatusResponse:
    r = self._request_raw(
      'DELETE',
      '/api/delete',
      json=DeleteRequest(
        model=model,
      ).model_dump(exclude_none=True),
    )
    return StatusResponse(
      status='success' if r.status_code == 200 else 'error',
    )

  def copy(self, source: str, destination: str) -> StatusResponse:
    r = self._request_raw(
      'POST',
      '/api/copy',
      json=CopyRequest(
        source=source,
        destination=destination,
      ).model_dump(exclude_none=True),
    )
    return StatusResponse(
      status='success' if r.status_code == 200 else 'error',
    )

  def show(self, model: str) -> ShowResponse:
    return self._request(
      ShowResponse,
      'POST',
      '/api/show',
      json=ShowRequest(
        model=model,
      ).model_dump(exclude_none=True),
    )

  def ps(self) -> ProcessResponse:
    return self._request(
      ProcessResponse,
      'GET',
      '/api/ps',
    )


class AsyncClient(BaseClient):
  def __init__(self, host: Optional[str] = None, **kwargs) -> None:
    super().__init__(httpx.AsyncClient, host, **kwargs)

  async def _request_raw(self, *args, **kwargs):
    r = await self._client.request(*args, **kwargs)
    try:
      r.raise_for_status()
    except httpx.HTTPStatusError as e:
      raise ResponseError(e.response.text, e.response.status_code) from None
    return r

  @overload
  async def _request(
    self,
    cls: Type[T],
    *args,
    stream: Literal[False] = False,
    **kwargs,
  ) -> T: ...

  @overload
  async def _request(
    self,
    cls: Type[T],
    *args,
    stream: Literal[True] = True,
    **kwargs,
  ) -> AsyncIterator[T]: ...

  @overload
  async def _request(
    self,
    cls: Type[T],
    *args,
    stream: bool = False,
    **kwargs,
  ) -> Union[T, AsyncIterator[T]]: ...

  async def _request(
    self,
    cls: Type[T],
    *args,
    stream: bool = False,
    **kwargs,
  ) -> Union[T, AsyncIterator[T]]:
    if stream:

      async def inner():
        async with self._client.stream(*args, **kwargs) as r:
          try:
            r.raise_for_status()
          except httpx.HTTPStatusError as e:
            await e.response.aread()
            raise ResponseError(e.response.text, e.response.status_code) from None

          async for line in r.aiter_lines():
            part = json.loads(line)
            if err := part.get('error'):
              raise ResponseError(err)
            yield cls(**part)

      return inner()

    return cls(**(await self._request_raw(*args, **kwargs)).json())

  @overload
  async def generate(
    self,
    model: str = '',
    prompt: str = '',
    suffix: str = '',
    *,
    system: str = '',
    template: str = '',
    context: Optional[Sequence[int]] = None,
    stream: Literal[False] = False,
    raw: bool = False,
    format: Optional[Literal['', 'json']] = None,
    images: Optional[Sequence[Union[str, bytes]]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> GenerateResponse: ...

  @overload
  async def generate(
    self,
    model: str = '',
    prompt: str = '',
    suffix: str = '',
    *,
    system: str = '',
    template: str = '',
    context: Optional[Sequence[int]] = None,
    stream: Literal[True] = True,
    raw: bool = False,
    format: Optional[Literal['', 'json']] = None,
    images: Optional[Sequence[Union[str, bytes]]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> AsyncIterator[GenerateResponse]: ...

  async def generate(
    self,
    model: str = '',
    prompt: Optional[str] = None,
    suffix: Optional[str] = None,
    *,
    system: Optional[str] = None,
    template: Optional[str] = None,
    context: Optional[Sequence[int]] = None,
    stream: bool = False,
    raw: Optional[bool] = None,
    format: Optional[Literal['', 'json']] = None,
    images: Optional[Sequence[Union[str, bytes]]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
    """
    Create a response using the requested model.

    Raises `RequestError` if a model is not provided.

    Raises `ResponseError` if the request could not be fulfilled.

    Returns `GenerateResponse` if `stream` is `False`, otherwise returns an asynchronous `GenerateResponse` generator.
    """
    return await self._request(
      GenerateResponse,
      'POST',
      '/api/generate',
      json=GenerateRequest(
        model=model,
        prompt=prompt,
        suffix=suffix,
        system=system,
        template=template,
        context=context,
        stream=stream,
        raw=raw,
        format=format,
        images=[Image(value=image) for image in images] if images else None,
        options=options,
        keep_alive=keep_alive,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  @overload
  async def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
    stream: Literal[False] = False,
    format: Optional[Literal['', 'json']] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> ChatResponse: ...

  @overload
  async def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
    stream: Literal[True] = True,
    format: Optional[Literal['', 'json']] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> AsyncIterator[ChatResponse]: ...

  async def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
    stream: bool = False,
    format: Optional[Literal['', 'json']] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
    """
    Create a chat response using the requested model.

    Args:
      tools:
        A JSON schema as a dict, an Ollama Tool or a Python Function.
        Python functions need to follow Google style docstrings to be converted to an Ollama Tool.
        For more information, see: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
      stream: Whether to stream the response.
      format: The format of the response.

    Example:
      def add_two_numbers(a: int, b: int) -> int:
        '''
        Add two numbers together.

        Args:
          a: First number to add
          b: Second number to add

        Returns:
          int: The sum of a and b
        '''
        return a + b

      await client.chat(model='llama3.2', tools=[add_two_numbers], messages=[...])

    Raises `RequestError` if a model is not provided.

    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ChatResponse` if `stream` is `False`, otherwise returns an asynchronous `ChatResponse` generator.
    """

    return await self._request(
      ChatResponse,
      'POST',
      '/api/chat',
      json=ChatRequest(
        model=model,
        messages=[message for message in _copy_messages(messages)],
        tools=[tool for tool in _copy_tools(tools)],
        stream=stream,
        format=format,
        options=options,
        keep_alive=keep_alive,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  async def embed(
    self,
    model: str = '',
    input: Union[str, Sequence[str]] = '',
    truncate: Optional[bool] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> EmbedResponse:
    return await self._request(
      EmbedResponse,
      'POST',
      '/api/embed',
      json=EmbedRequest(
        model=model,
        input=input,
        truncate=truncate,
        options=options,
        keep_alive=keep_alive,
      ).model_dump(exclude_none=True),
    )

  async def embeddings(
    self,
    model: str = '',
    prompt: Optional[str] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> EmbeddingsResponse:
    """
    Deprecated in favor of `embed`.
    """
    return await self._request(
      EmbeddingsResponse,
      'POST',
      '/api/embeddings',
      json=EmbeddingsRequest(
        model=model,
        prompt=prompt,
        options=options,
        keep_alive=keep_alive,
      ).model_dump(exclude_none=True),
    )

  @overload
  async def pull(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[False] = False,
  ) -> ProgressResponse: ...

  @overload
  async def pull(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[True] = True,
  ) -> AsyncIterator[ProgressResponse]: ...

  async def pull(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: bool = False,
  ) -> Union[ProgressResponse, AsyncIterator[ProgressResponse]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    return await self._request(
      ProgressResponse,
      'POST',
      '/api/pull',
      json=PullRequest(
        model=model,
        insecure=insecure,
        stream=stream,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  @overload
  async def push(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[False] = False,
  ) -> ProgressResponse: ...

  @overload
  async def push(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[True] = True,
  ) -> AsyncIterator[ProgressResponse]: ...

  async def push(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: bool = False,
  ) -> Union[ProgressResponse, AsyncIterator[ProgressResponse]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    return await self._request(
      ProgressResponse,
      'POST',
      '/api/push',
      json=PushRequest(
        model=model,
        insecure=insecure,
        stream=stream,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  @overload
  async def create(
    self,
    model: str,
    path: Optional[Union[str, PathLike]] = None,
    modelfile: Optional[str] = None,
    *,
    quantize: Optional[str] = None,
    stream: Literal[False] = False,
  ) -> ProgressResponse: ...

  @overload
  async def create(
    self,
    model: str,
    path: Optional[Union[str, PathLike]] = None,
    modelfile: Optional[str] = None,
    *,
    quantize: Optional[str] = None,
    stream: Literal[True] = True,
  ) -> AsyncIterator[ProgressResponse]: ...

  async def create(
    self,
    model: str,
    path: Optional[Union[str, PathLike]] = None,
    modelfile: Optional[str] = None,
    *,
    quantize: Optional[str] = None,
    stream: bool = False,
  ) -> Union[ProgressResponse, AsyncIterator[ProgressResponse]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    if (realpath := _as_path(path)) and realpath.exists():
      modelfile = await self._parse_modelfile(realpath.read_text(), base=realpath.parent)
    elif modelfile:
      modelfile = await self._parse_modelfile(modelfile)
    else:
      raise RequestError('must provide either path or modelfile')

    return await self._request(
      ProgressResponse,
      'POST',
      '/api/create',
      json=CreateRequest(
        model=model,
        modelfile=modelfile,
        stream=stream,
        quantize=quantize,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  async def _parse_modelfile(self, modelfile: str, base: Optional[Path] = None) -> str:
    base = Path.cwd() if base is None else base

    out = io.StringIO()
    for line in io.StringIO(modelfile):
      command, _, args = line.partition(' ')
      if command.upper() not in ['FROM', 'ADAPTER']:
        print(line, end='', file=out)
        continue

      path = Path(args.strip()).expanduser()
      path = path if path.is_absolute() else base / path
      if path.exists():
        args = f'@{await self._create_blob(path)}\n'
      print(command, args, end='', file=out)

    return out.getvalue()

  async def _create_blob(self, path: Union[str, Path]) -> str:
    sha256sum = sha256()
    with open(path, 'rb') as r:
      while True:
        chunk = r.read(32 * 1024)
        if not chunk:
          break
        sha256sum.update(chunk)

    digest = f'sha256:{sha256sum.hexdigest()}'

    async def upload_bytes():
      with open(path, 'rb') as r:
        while True:
          chunk = r.read(32 * 1024)
          if not chunk:
            break
          yield chunk

    await self._request_raw('POST', f'/api/blobs/{digest}', content=upload_bytes())

    return digest

  async def list(self) -> ListResponse:
    return await self._request(
      ListResponse,
      'GET',
      '/api/tags',
    )

  async def delete(self, model: str) -> StatusResponse:
    r = await self._request_raw(
      'DELETE',
      '/api/delete',
      json=DeleteRequest(
        model=model,
      ).model_dump(exclude_none=True),
    )
    return StatusResponse(
      status='success' if r.status_code == 200 else 'error',
    )

  async def copy(self, source: str, destination: str) -> StatusResponse:
    r = await self._request_raw(
      'POST',
      '/api/copy',
      json=CopyRequest(
        source=source,
        destination=destination,
      ).model_dump(exclude_none=True),
    )
    return StatusResponse(
      status='success' if r.status_code == 200 else 'error',
    )

  async def show(self, model: str) -> ShowResponse:
    return await self._request(
      ShowResponse,
      'POST',
      '/api/show',
      json=ShowRequest(
        model=model,
      ).model_dump(exclude_none=True),
    )

  async def ps(self) -> ProcessResponse:
    return await self._request(
      ProcessResponse,
      'GET',
      '/api/ps',
    )


def _copy_messages(messages: Optional[Sequence[Union[Mapping[str, Any], Message]]]) -> Iterator[Message]:
  for message in messages or []:
    yield Message.model_validate(
      {k: [Image(value=image) for image in v] if k == 'images' else v for k, v in dict(message).items() if v},
    )


def _copy_tools(tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None) -> Iterator[Tool]:
  for unprocessed_tool in tools or []:
    yield convert_function_to_tool(unprocessed_tool) if callable(unprocessed_tool) else Tool.model_validate(unprocessed_tool)


def _as_path(s: Optional[Union[str, PathLike]]) -> Union[Path, None]:
  if isinstance(s, str) or isinstance(s, Path):
    try:
      if (p := Path(s)).exists():
        return p
    except Exception:
      ...
  return None


def _parse_host(host: Optional[str]) -> str:
  """
  >>> _parse_host(None)
  'http://127.0.0.1:11434'
  >>> _parse_host('')
  'http://127.0.0.1:11434'
  >>> _parse_host('1.2.3.4')
  'http://1.2.3.4:11434'
  >>> _parse_host(':56789')
  'http://127.0.0.1:56789'
  >>> _parse_host('1.2.3.4:56789')
  'http://1.2.3.4:56789'
  >>> _parse_host('http://1.2.3.4')
  'http://1.2.3.4:80'
  >>> _parse_host('https://1.2.3.4')
  'https://1.2.3.4:443'
  >>> _parse_host('https://1.2.3.4:56789')
  'https://1.2.3.4:56789'
  >>> _parse_host('example.com')
  'http://example.com:11434'
  >>> _parse_host('example.com:56789')
  'http://example.com:56789'
  >>> _parse_host('http://example.com')
  'http://example.com:80'
  >>> _parse_host('https://example.com')
  'https://example.com:443'
  >>> _parse_host('https://example.com:56789')
  'https://example.com:56789'
  >>> _parse_host('example.com/')
  'http://example.com:11434'
  >>> _parse_host('example.com:56789/')
  'http://example.com:56789'
  >>> _parse_host('example.com/path')
  'http://example.com:11434/path'
  >>> _parse_host('example.com:56789/path')
  'http://example.com:56789/path'
  >>> _parse_host('https://example.com:56789/path')
  'https://example.com:56789/path'
  >>> _parse_host('example.com:56789/path/')
  'http://example.com:56789/path'
  >>> _parse_host('[0001:002:003:0004::1]')
  'http://[0001:002:003:0004::1]:11434'
  >>> _parse_host('[0001:002:003:0004::1]:56789')
  'http://[0001:002:003:0004::1]:56789'
  >>> _parse_host('http://[0001:002:003:0004::1]')
  'http://[0001:002:003:0004::1]:80'
  >>> _parse_host('https://[0001:002:003:0004::1]')
  'https://[0001:002:003:0004::1]:443'
  >>> _parse_host('https://[0001:002:003:0004::1]:56789')
  'https://[0001:002:003:0004::1]:56789'
  >>> _parse_host('[0001:002:003:0004::1]/')
  'http://[0001:002:003:0004::1]:11434'
  >>> _parse_host('[0001:002:003:0004::1]:56789/')
  'http://[0001:002:003:0004::1]:56789'
  >>> _parse_host('[0001:002:003:0004::1]/path')
  'http://[0001:002:003:0004::1]:11434/path'
  >>> _parse_host('[0001:002:003:0004::1]:56789/path')
  'http://[0001:002:003:0004::1]:56789/path'
  >>> _parse_host('https://[0001:002:003:0004::1]:56789/path')
  'https://[0001:002:003:0004::1]:56789/path'
  >>> _parse_host('[0001:002:003:0004::1]:56789/path/')
  'http://[0001:002:003:0004::1]:56789/path'
  """

  host, port = host or '', 11434
  scheme, _, hostport = host.partition('://')
  if not hostport:
    scheme, hostport = 'http', host
  elif scheme == 'http':
    port = 80
  elif scheme == 'https':
    port = 443

  split = urllib.parse.urlsplit('://'.join([scheme, hostport]))
  host = split.hostname or '127.0.0.1'
  port = split.port or port

  try:
    if isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address):
      # Fix missing square brackets for IPv6 from urlsplit
      host = f'[{host}]'
  except ValueError:
    ...

  if path := split.path.strip('/'):
    return f'{scheme}://{host}:{port}/{path}'

  return f'{scheme}://{host}:{port}'
