import io
import json
import httpx
from pathlib import Path
from hashlib import sha256
from base64 import b64encode


class BaseClient:

  def __init__(self, client, base_url='http://127.0.0.1:11434'):
    self._client = client(base_url=base_url, follow_redirects=True, timeout=None)


class Client(BaseClient):

  def __init__(self, base='http://localhost:11434'):
    super().__init__(httpx.Client, base)

  def _request(self, method, url, **kwargs):
    response = self._client.request(method, url, **kwargs)
    response.raise_for_status()
    return response

  def _request_json(self, method, url, **kwargs):
    return self._request(method, url, **kwargs).json()

  def _stream(self, method, url, **kwargs):
    with self._client.stream(method, url, **kwargs) as r:
      for line in r.iter_lines():
        part = json.loads(line)
        if e := part.get('error'):
          raise Exception(e)
        yield part

  def generate(self, model='', prompt='', system='', template='', context=None, stream=False, raw=False, format='', images=None, options=None):
    if not model:
      raise Exception('must provide a model')

    fn = self._stream if stream else self._request_json
    return fn('POST', '/api/generate', json={
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
    })

  def chat(self, model='', messages=None, stream=False, format='', options=None):
    if not model:
      raise Exception('must provide a model')

    for message in messages or []:
      if not isinstance(message, dict):
        raise TypeError('messages must be a list of strings')
      if not (role := message.get('role')) or role not in ['system', 'user', 'assistant']:
        raise Exception('messages must contain a role and it must be one of "system", "user", or "assistant"')
      if not message.get('content'):
        raise Exception('messages must contain content')
      if images := message.get('images'):
        message['images'] = [_encode_image(image) for image in images]

    fn = self._stream if stream else self._request_json
    return fn('POST', '/api/chat', json={
      'model': model,
      'messages': messages,
      'stream': stream,
      'format': format,
      'options': options or {},
    })

  def pull(self, model, insecure=False, stream=False):
    fn = self._stream if stream else self._request_json
    return fn('POST', '/api/pull', json={
      'model': model,
      'insecure': insecure,
      'stream': stream,
    })

  def push(self, model, insecure=False, stream=False):
    fn = self._stream if stream else self._request_json
    return fn('POST', '/api/push', json={
      'model': model,
      'insecure': insecure,
      'stream': stream,
    })

  def create(self, model, path=None, modelfile=None, stream=False):
    if (path := _as_path(path)) and path.exists():
      modelfile = self._parse_modelfile(path.read_text(), base=path.parent)
    elif modelfile:
      modelfile = self._parse_modelfile(modelfile)
    else:
      raise Exception('must provide either path or modelfile')

    fn = self._stream if stream else self._request_json
    return fn('POST', '/api/create', json={
      'model': model,
      'modelfile': modelfile,
      'stream': stream,
    })

  def _parse_modelfile(self, modelfile, base=None):
    base = Path.cwd() if base is None else base

    out = io.StringIO()
    for line in io.StringIO(modelfile):
      command, _, args = line.partition(' ')
      if command.upper() in ['FROM', 'ADAPTER']:
        path = Path(args).expanduser()
        path = path if path.is_absolute() else base / path
        if path.exists():
          args = f'@{self._create_blob(path)}'

      print(command, args, file=out)
    return out.getvalue()

  def _create_blob(self, path):
    sha256sum = sha256()
    with open(path, 'rb') as r:
      while True:
        chunk = r.read(32*1024)
        if not chunk:
          break
        sha256sum.update(chunk)

    digest = f'sha256:{sha256sum.hexdigest()}'

    try:
      self._request('HEAD', f'/api/blobs/{digest}')
    except httpx.HTTPStatusError as e:
      if e.response.status_code != 404:
        raise

      with open(path, 'rb') as r:
        self._request('PUT', f'/api/blobs/{digest}', content=r)

    return digest

  def delete(self, model):
    response = self._request_json('DELETE', '/api/delete', json={'model': model})
    return {'status': 'success' if response.status_code == 200 else 'error'}

  def list(self):
    return self._request_json('GET', '/api/tags').get('models', [])

  def copy(self, source, target):
    response = self._request_json('POST', '/api/copy', json={'source': source, 'destination': target})
    return {'status': 'success' if response.status_code == 200 else 'error'}

  def show(self, model):
    return self._request_json('GET', '/api/show', json={'model': model})


class AsyncClient(BaseClient):

  def __init__(self, base='http://localhost:11434'):
    super().__init__(httpx.AsyncClient, base)

  async def _request(self, method, url, **kwargs):
    response = await self._client.request(method, url, **kwargs)
    response.raise_for_status()
    return response

  async def _request_json(self, method, url, **kwargs):
    response = await self._request(method, url, **kwargs)
    return response.json()

  async def _stream(self, method, url, **kwargs):
    async def inner():
      async with self._client.stream(method, url, **kwargs) as r:
        async for line in r.aiter_lines():
          part = json.loads(line)
          if e := part.get('error'):
            raise Exception(e)
          yield part
    return inner()

  async def generate(self, model='', prompt='', system='', template='', context=None, stream=False, raw=False, format='', images=None, options=None):
    if not model:
      raise Exception('must provide a model')

    fn = self._stream if stream else self._request_json
    return await fn('POST', '/api/generate', json={
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
    })

  async def chat(self, model='', messages=None, stream=False, format='', options=None):
    if not model:
      raise Exception('must provide a model')

    for message in messages or []:
      if not isinstance(message, dict):
        raise TypeError('messages must be a list of strings')
      if not (role := message.get('role')) or role not in ['system', 'user', 'assistant']:
        raise Exception('messages must contain a role and it must be one of "system", "user", or "assistant"')
      if not message.get('content'):
        raise Exception('messages must contain content')
      if images := message.get('images'):
        message['images'] = [_encode_image(image) for image in images]

    fn = self._stream if stream else self._request_json
    return await fn('POST', '/api/chat', json={
      'model': model,
      'messages': messages,
      'stream': stream,
      'format': format,
      'options': options or {},
    })

  async def pull(self, model, insecure=False, stream=False):
    fn = self._stream if stream else self._request_json
    return await fn('POST', '/api/pull', json={
      'model': model,
      'insecure': insecure,
      'stream': stream,
    })

  async def push(self, model, insecure=False, stream=False):
    fn = self._stream if stream else self._request_json
    return await fn('POST', '/api/push', json={
      'model': model,
      'insecure': insecure,
      'stream': stream,
    })

  async def create(self, model, path=None, modelfile=None, stream=False):
    if (path := _as_path(path)) and path.exists():
      modelfile = await self._parse_modelfile(path.read_text(), base=path.parent)
    elif modelfile:
      modelfile = await self._parse_modelfile(modelfile)
    else:
      raise Exception('must provide either path or modelfile')

    fn = self._stream if stream else self._request_json
    return await fn('POST', '/api/create', json={
      'model': model,
      'modelfile': modelfile,
      'stream': stream,
    })

  async def _parse_modelfile(self, modelfile, base=None):
    base = Path.cwd() if base is None else base

    out = io.StringIO()
    for line in io.StringIO(modelfile):
      command, _, args = line.partition(' ')
      if command.upper() in ['FROM', 'ADAPTER']:
        path = Path(args).expanduser()
        path = path if path.is_absolute() else base / path
        if path.exists():
          args = f'@{await self._create_blob(path)}'

      print(command, args, file=out)
    return out.getvalue()

  async def _create_blob(self, path):
    sha256sum = sha256()
    with open(path, 'rb') as r:
      while True:
        chunk = r.read(32*1024)
        if not chunk:
          break
        sha256sum.update(chunk)

    digest = f'sha256:{sha256sum.hexdigest()}'

    try:
      await self._request('HEAD', f'/api/blobs/{digest}')
    except httpx.HTTPStatusError as e:
      if e.response.status_code != 404:
        raise

      async def upload_bytes():
        with open(path, 'rb') as r:
          while True:
            chunk = r.read(32*1024)
            if not chunk:
              break
            yield chunk

      await self._request('PUT', f'/api/blobs/{digest}', content=upload_bytes())

    return digest

  async def delete(self, model):
    response = await self._request_json('DELETE', '/api/delete', json={'model': model})
    return {'status': 'success' if response.status_code == 200 else 'error'}

  async def list(self):
    response = await self._request_json('GET', '/api/tags')
    return response.get('models', [])

  async def copy(self, source, target):
    response = await self._request_json('POST', '/api/copy', json={'source': source, 'destination': target})
    return {'status': 'success' if response.status_code == 200 else 'error'}

  async def show(self, model):
    return await self._request_json('GET', '/api/show', json={'model': model})


def _encode_image(image):
  if p := _as_path(image):
    b64 = b64encode(p.read_bytes())
  elif b := _as_bytesio(image):
    b64 = b64encode(b.read())
  else:
    raise Exception('images must be a list of bytes, path-like objects, or file-like objects')

  return b64.decode('utf-8')


def _as_path(s):
  if isinstance(s, str) or isinstance(s, Path):
    return Path(s)
  return None

def _as_bytesio(s):
  if isinstance(s, io.BytesIO):
    return s
  elif isinstance(s, bytes):
    return io.BytesIO(s)
  return None
