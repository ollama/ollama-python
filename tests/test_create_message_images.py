import base64
import copy
import json

import httpx
import pytest

from ollama import AsyncClient, Client, Image, Message

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC')


@pytest.fixture
def anyio_backend():
  return 'asyncio'


@pytest.mark.anyio
@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('kind', ['bytes', 'path', 'string_path', 'base64', 'image', 'message'])
async def test_create_normalizes_message_images(tmp_path, asynchronous, stream, kind):
  path = tmp_path / 'image.png'
  path.write_bytes(PNG)
  encoded = base64.b64encode(PNG).decode()
  value = {'bytes': PNG, 'path': path, 'string_path': str(path), 'base64': encoded, 'image': Image(value=PNG), 'message': Image(value=PNG)}[kind]
  message = Message(role='user', content='describe', images=[value]) if kind == 'message' else {'role': 'user', 'content': 'describe', 'images': [value]}
  original = copy.deepcopy(message)
  requests = []

  def handle(request):
    requests.append(json.loads(request.content))
    return httpx.Response(200, json={'status': 'success'})

  transport = httpx.MockTransport(handle)
  if asynchronous:
    async with AsyncClient(transport=transport) as client:
      result = await client.create('vision-example', messages=[message], stream=stream)
      statuses = [part.status async for part in result] if stream else [result.status]
  else:
    with Client(transport=transport) as client:
      result = client.create('vision-example', messages=[message], stream=stream)
      statuses = [part.status for part in result] if stream else [result.status]
  assert statuses == ['success']
  assert requests == [{'model': 'vision-example', 'stream': stream, 'messages': [{'role': 'user', 'content': 'describe', 'images': [encoded]}]}]
  assert message == original


@pytest.mark.anyio
@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('messages', [None, [], [{'role': 'user', 'content': '', 'custom': {'value': 0}}], [{'role': 'user', 'images': None}]])
async def test_create_preserves_message_payloads(asynchronous, messages):
  requests = []

  def handle(request):
    requests.append(json.loads(request.content))
    return httpx.Response(200, json={'status': 'success'})

  transport = httpx.MockTransport(handle)
  if asynchronous:
    async with AsyncClient(transport=transport) as client:
      await client.create('example', messages=messages)
  else:
    with Client(transport=transport) as client:
      client.create('example', messages=messages)
  expected = {'model': 'example', 'stream': False}
  if messages is not None:
    expected['messages'] = messages
  assert requests == [expected]
