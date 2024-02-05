import io
import json
import os
import sys
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import Any, AnyStr, Union, Optional, Sequence, Mapping, Literal

import httpx

from ollama._errors import ResponseError, RequestError
from ollama._utils import _as_path, _encode_image, _parse_host

if sys.version_info < (3, 9):
  from typing import Iterator, AsyncIterator
else:
  from collections.abc import Iterator, AsyncIterator

from ollama._types import Message, Options


class BaseClient:
  def __init__(
    self,
    client,
    host: Optional[str] = None,
    follow_redirects: bool = True,
    timeout: Any = None,
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
      **kwargs,
    )


class Client(BaseClient):
  def __init__(self, host: Optional[str] = None, **kwargs) -> None:
    super().__init__(httpx.Client, host, **kwargs)

  def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
    response = self._client.request(method, url, **kwargs)

    try:
      response.raise_for_status()
    except httpx.HTTPStatusError as e:
      raise ResponseError(e.response.text, e.response.status_code) from None

    return response

  def _stream(self, method: str, url: str, **kwargs) -> Iterator[Mapping[str, Any]]:
    with self._client.stream(method, url, **kwargs) as r:
      try:
        r.raise_for_status()
      except httpx.HTTPStatusError as e:
        e.response.read()
        raise ResponseError(e.response.text, e.response.status_code) from None

      for line in r.iter_lines():
        partial = json.loads(line)
        if e := partial.get('error'):
          raise ResponseError(e)
        yield partial

  def _request_stream(
    self,
    *args,
    stream: bool = False,
    **kwargs,
  ) -> Union[Mapping[str, Any], Iterator[Mapping[str, Any]]]:
    return self._stream(*args, **kwargs) if stream else self._request(*args, **kwargs).json()

  def generate(
    self,
    model: str = '',
    prompt: str = '',
    system: str = '',
    template: str = '',
    context: Optional[Sequence[int]] = None,
    stream: bool = False,
    raw: bool = False,
    format: Literal['', 'json'] = '',
    images: Optional[Sequence[AnyStr]] = None,
    options: Optional[Options] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Union[Mapping[str, Any], Iterator[Mapping[str, Any]]]:
    """
    Create a response using the requested model.

    Raises `RequestError` if a model is not provided.

    Raises `ResponseError` if the request could not be fulfilled.

    Returns `GenerateResponse` if `stream` is `False`, otherwise returns a `GenerateResponse` generator.
    """

    if not model:
      raise RequestError('must provide a model')

    return self._request_stream(
      'POST',
      '/api/generate',
      json={
        'model': model,
        'prompt': prompt,
        'system': system,
        'template': template,
        'context': context or [],
        'stream': stream,
        'raw': raw,
        'images': [_encode_image(image) for image in images or []],
        'format': format,
        'options': options or {},
        'keep_alive': keep_alive,
      },
      stream=stream,
    )

  def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Message]] = None,
    stream: bool = False,
    format: Literal['', 'json'] = '',
    options: Optional[Options] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Union[Mapping[str, Any], Iterator[Mapping[str, Any]]]:
    """
    Create a chat response using the requested model.

    Raises `RequestError` if a model is not provided.

    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ChatResponse` if `stream` is `False`, otherwise returns a `ChatResponse` generator.
    """

    if not model:
      raise RequestError('must provide a model')

    return self._request_stream(
      'POST',
      '/api/chat',
      json={
        'model': model,
        'messages': [message.dict() for message in messages],
        'stream': stream,
        'format': format,
        'options': options or {},
        'keep_alive': keep_alive,
      },
      stream=stream,
    )

  def embeddings(
    self,
    model: str = '',
    prompt: str = '',
    options: Optional[Options] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Sequence[float]:
    return self._request(
      'POST',
      '/api/embeddings',
      json={
        'model': model,
        'prompt': prompt,
        'options': options or {},
        'keep_alive': keep_alive,
      },
    ).json()

  def pull(
    self,
    model: str,
    insecure: bool = False,
    stream: bool = False,
  ) -> Union[Mapping[str, Any], Iterator[Mapping[str, Any]]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    return self._request_stream(
      'POST',
      '/api/pull',
      json={
        'name': model,
        'insecure': insecure,
        'stream': stream,
      },
      stream=stream,
    )

  def push(
    self,
    model: str,
    insecure: bool = False,
    stream: bool = False,
  ) -> Union[Mapping[str, Any], Iterator[Mapping[str, Any]]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    return self._request_stream(
      'POST',
      '/api/push',
      json={
        'name': model,
        'insecure': insecure,
        'stream': stream,
      },
      stream=stream,
    )

  def create(
    self,
    model: str,
    path: Optional[Union[str, PathLike]] = None,
    modelfile: Optional[str] = None,
    stream: bool = False,
  ) -> Union[Mapping[str, Any], Iterator[Mapping[str, Any]]]:
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

    return self._request_stream(
      'POST',
      '/api/create',
      json={
        'name': model,
        'modelfile': modelfile,
        'stream': stream,
      },
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

    try:
      self._request('HEAD', f'/api/blobs/{digest}')
    except ResponseError as e:
      if e.status_code != 404:
        raise

      with open(path, 'rb') as r:
        self._request('POST', f'/api/blobs/{digest}', content=r)

    return digest

  def delete(self, model: str) -> Mapping[str, Any]:
    response = self._request('DELETE', '/api/delete', json={'name': model})
    return {'status': 'success' if response.status_code == 200 else 'error'}

  def list(self) -> Mapping[str, Any]:
    return self._request('GET', '/api/tags').json()

  def copy(self, source: str, destination: str) -> Mapping[str, Any]:
    response = self._request('POST', '/api/copy', json={'source': source, 'destination': destination})
    return {'status': 'success' if response.status_code == 200 else 'error'}

  def show(self, model: str) -> Mapping[str, Any]:
    return self._request('POST', '/api/show', json={'name': model}).json()


class AsyncClient(BaseClient):
  def __init__(self, host: Optional[str] = None, **kwargs) -> None:
    super().__init__(httpx.AsyncClient, host, **kwargs)

  async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
    response = await self._client.request(method, url, **kwargs)

    try:
      response.raise_for_status()
    except httpx.HTTPStatusError as e:
      raise ResponseError(e.response.text, e.response.status_code) from None

    return response

  async def _stream(self, method: str, url: str, **kwargs) -> AsyncIterator[Mapping[str, Any]]:
    async def inner():
      async with self._client.stream(method, url, **kwargs) as r:
        try:
          r.raise_for_status()
        except httpx.HTTPStatusError as e:
          e.response.read()
          raise ResponseError(e.response.text, e.response.status_code) from None

        async for line in r.aiter_lines():
          partial = json.loads(line)
          if e := partial.get('error'):
            raise ResponseError(e)
          yield partial

    return inner()

  async def _request_stream(
    self,
    *args,
    stream: bool = False,
    **kwargs,
  ) -> Union[Mapping[str, Any], AsyncIterator[Mapping[str, Any]]]:
    if stream:
      return await self._stream(*args, **kwargs)

    response = await self._request(*args, **kwargs)
    return response.json()

  async def generate(
    self,
    model: str = '',
    prompt: str = '',
    system: str = '',
    template: str = '',
    context: Optional[Sequence[int]] = None,
    stream: bool = False,
    raw: bool = False,
    format: Literal['', 'json'] = '',
    images: Optional[Sequence[AnyStr]] = None,
    options: Optional[Options] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Union[Mapping[str, Any], AsyncIterator[Mapping[str, Any]]]:
    """
    Create a response using the requested model.

    Raises `RequestError` if a model is not provided.

    Raises `ResponseError` if the request could not be fulfilled.

    Returns `GenerateResponse` if `stream` is `False`, otherwise returns an asynchronous `GenerateResponse` generator.
    """
    if not model:
      raise RequestError('must provide a model')

    return await self._request_stream(
      'POST',
      '/api/generate',
      json={
        'model': model,
        'prompt': prompt,
        'system': system,
        'template': template,
        'context': context or [],
        'stream': stream,
        'raw': raw,
        'images': [_encode_image(image) for image in images or []],
        'format': format,
        'options': options or {},
        'keep_alive': keep_alive,
      },
      stream=stream,
    )

  async def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Message]] = None,
    stream: bool = False,
    format: Literal['', 'json'] = '',
    options: Optional[Options] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Union[Mapping[str, Any], AsyncIterator[Mapping[str, Any]]]:
    """
    Create a chat response using the requested model.

    Raises `RequestError` if a model is not provided.

    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ChatResponse` if `stream` is `False`, otherwise returns an asynchronous `ChatResponse` generator.
    """
    if not model:
      raise RequestError('must provide a model')

    return await self._request_stream(
      'POST',
      '/api/chat',
      json={
        'model': model,
        'messages': messages,
        'stream': stream,
        'format': format,
        'options': options or {},
        'keep_alive': keep_alive,
      },
      stream=stream,
    )

  async def embeddings(
    self,
    model: str = '',
    prompt: str = '',
    options: Optional[Options] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Sequence[float]:
    response = await self._request(
      'POST',
      '/api/embeddings',
      json={
        'model': model,
        'prompt': prompt,
        'options': options or {},
        'keep_alive': keep_alive,
      },
    )

    return response.json()

  async def pull(
    self,
    model: str,
    insecure: bool = False,
    stream: bool = False,
  ) -> Union[Mapping[str, Any], AsyncIterator[Mapping[str, Any]]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    return await self._request_stream(
      'POST',
      '/api/pull',
      json={
        'name': model,
        'insecure': insecure,
        'stream': stream,
      },
      stream=stream,
    )

  async def push(
    self,
    model: str,
    insecure: bool = False,
    stream: bool = False,
  ) -> Union[Mapping[str, Any], AsyncIterator[Mapping[str, Any]]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    return await self._request_stream(
      'POST',
      '/api/push',
      json={
        'name': model,
        'insecure': insecure,
        'stream': stream,
      },
      stream=stream,
    )

  async def create(
    self,
    model: str,
    path: Optional[Union[str, PathLike]] = None,
    modelfile: Optional[str] = None,
    stream: bool = False,
  ) -> Union[Mapping[str, Any], AsyncIterator[Mapping[str, Any]]]:
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

    return await self._request_stream(
      'POST',
      '/api/create',
      json={
        'name': model,
        'modelfile': modelfile,
        'stream': stream,
      },
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

    try:
      await self._request('HEAD', f'/api/blobs/{digest}')
    except ResponseError as e:
      if e.status_code != 404:
        raise

      async def upload_bytes():
        with open(path, 'rb') as r:
          while True:
            chunk = r.read(32 * 1024)
            if not chunk:
              break
            yield chunk

      await self._request('POST', f'/api/blobs/{digest}', content=upload_bytes())

    return digest

  async def delete(self, model: str) -> Mapping[str, Any]:
    response = await self._request('DELETE', '/api/delete', json={'name': model})
    return {'status': 'success' if response.status_code == 200 else 'error'}

  async def list(self) -> Mapping[str, Any]:
    response = await self._request('GET', '/api/tags')
    return response.json()

  async def copy(self, source: str, destination: str) -> Mapping[str, Any]:
    response = await self._request('POST', '/api/copy', json={'source': source, 'destination': destination})
    return {'status': 'success' if response.status_code == 200 else 'error'}

  async def show(self, model: str) -> Mapping[str, Any]:
    response = await self._request('POST', '/api/show', json={'name': model})
    return response.json()
