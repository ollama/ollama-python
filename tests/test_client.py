import base64
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest
from httpx import Response as httpxResponse
from pydantic import BaseModel
from pytest_httpserver import HTTPServer, URIPattern
from werkzeug.wrappers import Request, Response

from ollama._client import CONNECTION_ERROR_MESSAGE, AsyncClient, Client, _copy_tools
from ollama._types import Image, Message

PNG_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC'
PNG_BYTES = base64.b64decode(PNG_BASE64)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
  return 'asyncio'


class PrefixPattern(URIPattern):
  def __init__(self, prefix: str):
    self.prefix = prefix

  def match(self, uri):
    return uri.startswith(self.prefix)


def test_client_chat(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/chat',
    method='POST',
    json={
      'model': 'dummy',
      'messages': [{'role': 'user', 'content': 'Why is the sky blue?'}],
      'tools': [],
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': "I don't know.",
      },
    }
  )

  client = Client(httpserver.url_for('/'))
  response = client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
  assert response['model'] == 'dummy'
  assert response['message']['role'] == 'assistant'
  assert response['message']['content'] == "I don't know."


def test_client_chat_stream(httpserver: HTTPServer):
  def stream_handler(_: Request):
    def generate():
      for message in ['I ', "don't ", 'know.']:
        yield (
          json.dumps(
            {
              'model': 'dummy',
              'message': {
                'role': 'assistant',
                'content': message,
              },
            }
          )
          + '\n'
        )

    return Response(generate())

  httpserver.expect_ordered_request(
    '/api/chat',
    method='POST',
    json={
      'model': 'dummy',
      'messages': [{'role': 'user', 'content': 'Why is the sky blue?'}],
      'tools': [],
      'stream': True,
    },
  ).respond_with_handler(stream_handler)

  client = Client(httpserver.url_for('/'))
  response = client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}], stream=True)

  it = iter(['I ', "don't ", 'know.'])
  for part in response:
    assert part['message']['role'] in 'assistant'
    assert part['message']['content'] == next(it)


@pytest.mark.parametrize('message_format', ('dict', 'pydantic_model'))
@pytest.mark.parametrize('file_style', ('path', 'bytes'))
def test_client_chat_images(httpserver: HTTPServer, message_format: str, file_style: str, tmp_path):
  from ollama._types import Image, Message

  httpserver.expect_ordered_request(
    '/api/chat',
    method='POST',
    json={
      'model': 'dummy',
      'messages': [
        {
          'role': 'user',
          'content': 'Why is the sky blue?',
          'images': [PNG_BASE64],
        },
      ],
      'tools': [],
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': "I don't know.",
      },
    }
  )

  client = Client(httpserver.url_for('/'))

  if file_style == 'bytes':
    image_content = PNG_BYTES
  elif file_style == 'path':
    image_path = tmp_path / 'transparent.png'
    image_path.write_bytes(PNG_BYTES)
    image_content = str(image_path)

  if message_format == 'pydantic_model':
    messages = [Message(role='user', content='Why is the sky blue?', images=[Image(value=image_content)])]
  elif message_format == 'dict':
    messages = [{'role': 'user', 'content': 'Why is the sky blue?', 'images': [image_content]}]
  else:
    raise ValueError(f'Invalid message format: {message_format}')

  response = client.chat('dummy', messages=messages)
  assert response['model'] == 'dummy'
  assert response['message']['role'] == 'assistant'
  assert response['message']['content'] == "I don't know."


def test_client_chat_format_json(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/chat',
    method='POST',
    json={
      'model': 'dummy',
      'messages': [{'role': 'user', 'content': 'Why is the sky blue?'}],
      'tools': [],
      'format': 'json',
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': '{"answer": "Because of Rayleigh scattering"}',
      },
    }
  )

  client = Client(httpserver.url_for('/'))
  response = client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}], format='json')
  assert response['model'] == 'dummy'
  assert response['message']['role'] == 'assistant'
  assert response['message']['content'] == '{"answer": "Because of Rayleigh scattering"}'


def test_client_chat_format_pydantic(httpserver: HTTPServer):
  class ResponseFormat(BaseModel):
    answer: str
    confidence: float

  httpserver.expect_ordered_request(
    '/api/chat',
    method='POST',
    json={
      'model': 'dummy',
      'messages': [{'role': 'user', 'content': 'Why is the sky blue?'}],
      'tools': [],
      'format': {'title': 'ResponseFormat', 'type': 'object', 'properties': {'answer': {'title': 'Answer', 'type': 'string'}, 'confidence': {'title': 'Confidence', 'type': 'number'}}, 'required': ['answer', 'confidence']},
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': '{"answer": "Because of Rayleigh scattering", "confidence": 0.95}',
      },
    }
  )

  client = Client(httpserver.url_for('/'))
  response = client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}], format=ResponseFormat.model_json_schema())
  assert response['model'] == 'dummy'
  assert response['message']['role'] == 'assistant'
  assert response['message']['content'] == '{"answer": "Because of Rayleigh scattering", "confidence": 0.95}'


async def test_async_client_chat_format_json(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/chat',
    method='POST',
    json={
      'model': 'dummy',
      'messages': [{'role': 'user', 'content': 'Why is the sky blue?'}],
      'tools': [],
      'format': 'json',
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': '{"answer": "Because of Rayleigh scattering"}',
      },
    }
  )

  client = AsyncClient(httpserver.url_for('/'))
  response = await client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}], format='json')
  assert response['model'] == 'dummy'
  assert response['message']['role'] == 'assistant'
  assert response['message']['content'] == '{"answer": "Because of Rayleigh scattering"}'


async def test_async_client_chat_format_pydantic(httpserver: HTTPServer):
  class ResponseFormat(BaseModel):
    answer: str
    confidence: float

  httpserver.expect_ordered_request(
    '/api/chat',
    method='POST',
    json={
      'model': 'dummy',
      'messages': [{'role': 'user', 'content': 'Why is the sky blue?'}],
      'tools': [],
      'format': {'title': 'ResponseFormat', 'type': 'object', 'properties': {'answer': {'title': 'Answer', 'type': 'string'}, 'confidence': {'title': 'Confidence', 'type': 'number'}}, 'required': ['answer', 'confidence']},
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': '{"answer": "Because of Rayleigh scattering", "confidence": 0.95}',
      },
    }
  )

  client = AsyncClient(httpserver.url_for('/'))
  response = await client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}], format=ResponseFormat.model_json_schema())
  assert response['model'] == 'dummy'
  assert response['message']['role'] == 'assistant'
  assert response['message']['content'] == '{"answer": "Because of Rayleigh scattering", "confidence": 0.95}'


def test_client_generate(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/generate',
    method='POST',
    json={
      'model': 'dummy',
      'prompt': 'Why is the sky blue?',
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'response': 'Because it is.',
    }
  )

  client = Client(httpserver.url_for('/'))
  response = client.generate('dummy', 'Why is the sky blue?')
  assert response['model'] == 'dummy'
  assert response['response'] == 'Because it is.'


def test_client_generate_with_image_type(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/generate',
    method='POST',
    json={
      'model': 'dummy',
      'prompt': 'What is in this image?',
      'stream': False,
      'images': [PNG_BASE64],
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'response': 'A blue sky.',
    }
  )

  client = Client(httpserver.url_for('/'))
  response = client.generate('dummy', 'What is in this image?', images=[Image(value=PNG_BASE64)])
  assert response['model'] == 'dummy'
  assert response['response'] == 'A blue sky.'


def test_client_generate_with_invalid_image(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/generate',
    method='POST',
    json={
      'model': 'dummy',
      'prompt': 'What is in this image?',
      'stream': False,
      'images': ['invalid_base64'],
    },
  ).respond_with_json({'error': 'Invalid image data'}, status=400)

  client = Client(httpserver.url_for('/'))
  with pytest.raises(ValueError):
    client.generate('dummy', 'What is in this image?', images=[Image(value='invalid_base64')])


def test_client_generate_stream(httpserver: HTTPServer):
  def stream_handler(_: Request):
    def generate():
      for message in ['Because ', 'it ', 'is.']:
        yield (
          json.dumps(
            {
              'model': 'dummy',
              'response': message,
            }
          )
          + '\n'
        )

    return Response(generate())

  httpserver.expect_ordered_request(
    '/api/generate',
    method='POST',
    json={
      'model': 'dummy',
      'prompt': 'Why is the sky blue?',
      'stream': True,
    },
  ).respond_with_handler(stream_handler)

  client = Client(httpserver.url_for('/'))
  response = client.generate('dummy', 'Why is the sky blue?', stream=True)

  it = iter(['Because ', 'it ', 'is.'])
  for part in response:
    assert part['model'] == 'dummy'
    assert part['response'] == next(it)


def test_client_generate_images(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/generate',
    method='POST',
    json={
      'model': 'dummy',
      'prompt': 'Why is the sky blue?',
      'stream': False,
      'images': [PNG_BASE64],
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'response': 'Because it is.',
    }
  )

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as temp:
    temp.write(PNG_BYTES)
    temp.flush()
    response = client.generate('dummy', 'Why is the sky blue?', images=[temp.name])
    assert response['model'] == 'dummy'
    assert response['response'] == 'Because it is.'


def test_client_generate_format_json(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/generate',
    method='POST',
    json={
      'model': 'dummy',
      'prompt': 'Why is the sky blue?',
      'format': 'json',
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'response': '{"answer": "Because of Rayleigh scattering"}',
    }
  )

  client = Client(httpserver.url_for('/'))
  response = client.generate('dummy', 'Why is the sky blue?', format='json')
  assert response['model'] == 'dummy'
  assert response['response'] == '{"answer": "Because of Rayleigh scattering"}'


def test_client_generate_format_pydantic(httpserver: HTTPServer):
  class ResponseFormat(BaseModel):
    answer: str
    confidence: float

  httpserver.expect_ordered_request(
    '/api/generate',
    method='POST',
    json={
      'model': 'dummy',
      'prompt': 'Why is the sky blue?',
      'format': {'title': 'ResponseFormat', 'type': 'object', 'properties': {'answer': {'title': 'Answer', 'type': 'string'}, 'confidence': {'title': 'Confidence', 'type': 'number'}}, 'required': ['answer', 'confidence']},
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'response': '{"answer": "Because of Rayleigh scattering", "confidence": 0.95}',
    }
  )

  client = Client(httpserver.url_for('/'))
  response = client.generate('dummy', 'Why is the sky blue?', format=ResponseFormat.model_json_schema())
  assert response['model'] == 'dummy'
  assert response['response'] == '{"answer": "Because of Rayleigh scattering", "confidence": 0.95}'


async def test_async_client_generate_format_json(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/generate',
    method='POST',
    json={
      'model': 'dummy',
      'prompt': 'Why is the sky blue?',
      'format': 'json',
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'response': '{"answer": "Because of Rayleigh scattering"}',
    }
  )

  client = AsyncClient(httpserver.url_for('/'))
  response = await client.generate('dummy', 'Why is the sky blue?', format='json')
  assert response['model'] == 'dummy'
  assert response['response'] == '{"answer": "Because of Rayleigh scattering"}'


async def test_async_client_generate_format_pydantic(httpserver: HTTPServer):
  class ResponseFormat(BaseModel):
    answer: str
    confidence: float

  httpserver.expect_ordered_request(
    '/api/generate',
    method='POST',
    json={
      'model': 'dummy',
      'prompt': 'Why is the sky blue?',
      'format': {'title': 'ResponseFormat', 'type': 'object', 'properties': {'answer': {'title': 'Answer', 'type': 'string'}, 'confidence': {'title': 'Confidence', 'type': 'number'}}, 'required': ['answer', 'confidence']},
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'response': '{"answer": "Because of Rayleigh scattering", "confidence": 0.95}',
    }
  )

  client = AsyncClient(httpserver.url_for('/'))
  response = await client.generate('dummy', 'Why is the sky blue?', format=ResponseFormat.model_json_schema())
  assert response['model'] == 'dummy'
  assert response['response'] == '{"answer": "Because of Rayleigh scattering", "confidence": 0.95}'


def test_client_pull(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/pull',
    method='POST',
    json={
      'model': 'dummy',
      'insecure': False,
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = Client(httpserver.url_for('/'))
  response = client.pull('dummy')
  assert response['status'] == 'success'


def test_client_pull_stream(httpserver: HTTPServer):
  def stream_handler(_: Request):
    def generate():
      yield json.dumps({'status': 'pulling manifest'}) + '\n'
      yield json.dumps({'status': 'verifying sha256 digest'}) + '\n'
      yield json.dumps({'status': 'writing manifest'}) + '\n'
      yield json.dumps({'status': 'removing any unused layers'}) + '\n'
      yield json.dumps({'status': 'success'}) + '\n'

    return Response(generate())

  httpserver.expect_ordered_request(
    '/api/pull',
    method='POST',
    json={
      'model': 'dummy',
      'insecure': False,
      'stream': True,
    },
  ).respond_with_handler(stream_handler)

  client = Client(httpserver.url_for('/'))
  response = client.pull('dummy', stream=True)

  it = iter(['pulling manifest', 'verifying sha256 digest', 'writing manifest', 'removing any unused layers', 'success'])
  for part in response:
    assert part['status'] == next(it)


def test_client_push(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/push',
    method='POST',
    json={
      'model': 'dummy',
      'insecure': False,
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = Client(httpserver.url_for('/'))
  response = client.push('dummy')
  assert response['status'] == 'success'


def test_client_push_stream(httpserver: HTTPServer):
  def stream_handler(_: Request):
    def generate():
      yield json.dumps({'status': 'retrieving manifest'}) + '\n'
      yield json.dumps({'status': 'pushing manifest'}) + '\n'
      yield json.dumps({'status': 'success'}) + '\n'

    return Response(generate())

  httpserver.expect_ordered_request(
    '/api/push',
    method='POST',
    json={
      'model': 'dummy',
      'insecure': False,
      'stream': True,
    },
  ).respond_with_handler(stream_handler)

  client = Client(httpserver.url_for('/'))
  response = client.push('dummy', stream=True)

  it = iter(['retrieving manifest', 'pushing manifest', 'success'])
  for part in response:
    assert part['status'] == next(it)


@pytest.fixture
def userhomedir():
  with tempfile.TemporaryDirectory() as temp:
    home = os.getenv('HOME', '')
    os.environ['HOME'] = temp
    yield Path(temp)
    os.environ['HOME'] = home


def test_client_create_with_blob(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'files': {'test.gguf': 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'},
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile():
    response = client.create('dummy', files={'test.gguf': 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'})
    assert response['status'] == 'success'


def test_client_create_with_parameters_roundtrip(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'quantize': 'q4_k_m',
      'from': 'mymodel',
      'adapters': {'someadapter.gguf': 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'},
      'template': '[INST] <<SYS>>{{.System}}<</SYS>>\n{{.Prompt}} [/INST]',
      'license': 'this is my license',
      'system': '\nUse\nmultiline\nstrings.\n',
      'parameters': {'stop': ['[INST]', '[/INST]', '<<SYS>>', '<</SYS>>'], 'pi': 3.14159},
      'messages': [{'role': 'user', 'content': 'Hello there!'}, {'role': 'assistant', 'content': 'Hello there yourself!'}],
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile():
    response = client.create(
      'dummy',
      quantize='q4_k_m',
      from_='mymodel',
      adapters={'someadapter.gguf': 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'},
      template='[INST] <<SYS>>{{.System}}<</SYS>>\n{{.Prompt}} [/INST]',
      license='this is my license',
      system='\nUse\nmultiline\nstrings.\n',
      parameters={'stop': ['[INST]', '[/INST]', '<<SYS>>', '<</SYS>>'], 'pi': 3.14159},
      messages=[{'role': 'user', 'content': 'Hello there!'}, {'role': 'assistant', 'content': 'Hello there yourself!'}],
      stream=False,
    )
    assert response['status'] == 'success'


def test_client_create_from_library(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'from': 'llama2',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = Client(httpserver.url_for('/'))

  response = client.create('dummy', from_='llama2')
  assert response['status'] == 'success'


def test_client_create_blob(httpserver: HTTPServer):
  httpserver.expect_ordered_request(re.compile('^/api/blobs/sha256[:-][0-9a-fA-F]{64}$'), method='POST').respond_with_response(Response(status=201))

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = client.create_blob(blob.name)
    assert response == 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


def test_client_create_blob_exists(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = client.create_blob(blob.name)
    assert response == 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


def test_client_delete(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/delete'), method='DELETE').respond_with_response(Response(status=200))
  client = Client(httpserver.url_for('/api/delete'))
  response = client.delete('dummy')
  assert response['status'] == 'success'


def test_client_copy(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/copy'), method='POST').respond_with_response(Response(status=200))
  client = Client(httpserver.url_for('/api/copy'))
  response = client.copy('dum', 'dummer')
  assert response['status'] == 'success'


async def test_async_client_chat(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/chat',
    method='POST',
    json={
      'model': 'dummy',
      'messages': [{'role': 'user', 'content': 'Why is the sky blue?'}],
      'tools': [],
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': "I don't know.",
      },
    }
  )

  client = AsyncClient(httpserver.url_for('/'))
  response = await client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
  assert response['model'] == 'dummy'
  assert response['message']['role'] == 'assistant'
  assert response['message']['content'] == "I don't know."


async def test_async_client_chat_stream(httpserver: HTTPServer):
  def stream_handler(_: Request):
    def generate():
      for message in ['I ', "don't ", 'know.']:
        yield (
          json.dumps(
            {
              'model': 'dummy',
              'message': {
                'role': 'assistant',
                'content': message,
              },
            }
          )
          + '\n'
        )

    return Response(generate())

  httpserver.expect_ordered_request(
    '/api/chat',
    method='POST',
    json={
      'model': 'dummy',
      'messages': [{'role': 'user', 'content': 'Why is the sky blue?'}],
      'tools': [],
      'stream': True,
    },
  ).respond_with_handler(stream_handler)

  client = AsyncClient(httpserver.url_for('/'))
  response = await client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}], stream=True)

  it = iter(['I ', "don't ", 'know.'])
  async for part in response:
    assert part['message']['role'] == 'assistant'
    assert part['message']['content'] == next(it)


async def test_async_client_chat_images(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/chat',
    method='POST',
    json={
      'model': 'dummy',
      'messages': [
        {
          'role': 'user',
          'content': 'Why is the sky blue?',
          'images': [PNG_BASE64],
        },
      ],
      'tools': [],
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': "I don't know.",
      },
    }
  )

  client = AsyncClient(httpserver.url_for('/'))

  response = await client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?', 'images': [PNG_BYTES]}])
  assert response['model'] == 'dummy'
  assert response['message']['role'] == 'assistant'
  assert response['message']['content'] == "I don't know."


async def test_async_client_generate(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/generate',
    method='POST',
    json={
      'model': 'dummy',
      'prompt': 'Why is the sky blue?',
      'stream': False,
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'response': 'Because it is.',
    }
  )

  client = AsyncClient(httpserver.url_for('/'))
  response = await client.generate('dummy', 'Why is the sky blue?')
  assert response['model'] == 'dummy'
  assert response['response'] == 'Because it is.'


async def test_async_client_generate_stream(httpserver: HTTPServer):
  def stream_handler(_: Request):
    def generate():
      for message in ['Because ', 'it ', 'is.']:
        yield (
          json.dumps(
            {
              'model': 'dummy',
              'response': message,
            }
          )
          + '\n'
        )

    return Response(generate())

  httpserver.expect_ordered_request(
    '/api/generate',
    method='POST',
    json={
      'model': 'dummy',
      'prompt': 'Why is the sky blue?',
      'stream': True,
    },
  ).respond_with_handler(stream_handler)

  client = AsyncClient(httpserver.url_for('/'))
  response = await client.generate('dummy', 'Why is the sky blue?', stream=True)

  it = iter(['Because ', 'it ', 'is.'])
  async for part in response:
    assert part['model'] == 'dummy'
    assert part['response'] == next(it)


async def test_async_client_generate_images(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/generate',
    method='POST',
    json={
      'model': 'dummy',
      'prompt': 'Why is the sky blue?',
      'stream': False,
      'images': [PNG_BASE64],
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'response': 'Because it is.',
    }
  )

  client = AsyncClient(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as temp:
    temp.write(PNG_BYTES)
    temp.flush()
    response = await client.generate('dummy', 'Why is the sky blue?', images=[temp.name])
    assert response['model'] == 'dummy'
    assert response['response'] == 'Because it is.'


async def test_async_client_pull(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/pull',
    method='POST',
    json={
      'model': 'dummy',
      'insecure': False,
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = AsyncClient(httpserver.url_for('/'))
  response = await client.pull('dummy')
  assert response['status'] == 'success'


async def test_async_client_pull_stream(httpserver: HTTPServer):
  def stream_handler(_: Request):
    def generate():
      yield json.dumps({'status': 'pulling manifest'}) + '\n'
      yield json.dumps({'status': 'verifying sha256 digest'}) + '\n'
      yield json.dumps({'status': 'writing manifest'}) + '\n'
      yield json.dumps({'status': 'removing any unused layers'}) + '\n'
      yield json.dumps({'status': 'success'}) + '\n'

    return Response(generate())

  httpserver.expect_ordered_request(
    '/api/pull',
    method='POST',
    json={
      'model': 'dummy',
      'insecure': False,
      'stream': True,
    },
  ).respond_with_handler(stream_handler)

  client = AsyncClient(httpserver.url_for('/'))
  response = await client.pull('dummy', stream=True)

  it = iter(['pulling manifest', 'verifying sha256 digest', 'writing manifest', 'removing any unused layers', 'success'])
  async for part in response:
    assert part['status'] == next(it)


async def test_async_client_push(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/push',
    method='POST',
    json={
      'model': 'dummy',
      'insecure': False,
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = AsyncClient(httpserver.url_for('/'))
  response = await client.push('dummy')
  assert response['status'] == 'success'


async def test_async_client_push_stream(httpserver: HTTPServer):
  def stream_handler(_: Request):
    def generate():
      yield json.dumps({'status': 'retrieving manifest'}) + '\n'
      yield json.dumps({'status': 'pushing manifest'}) + '\n'
      yield json.dumps({'status': 'success'}) + '\n'

    return Response(generate())

  httpserver.expect_ordered_request(
    '/api/push',
    method='POST',
    json={
      'model': 'dummy',
      'insecure': False,
      'stream': True,
    },
  ).respond_with_handler(stream_handler)

  client = AsyncClient(httpserver.url_for('/'))
  response = await client.push('dummy', stream=True)

  it = iter(['retrieving manifest', 'pushing manifest', 'success'])
  async for part in response:
    assert part['status'] == next(it)


async def test_async_client_create_with_blob(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'files': {'test.gguf': 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'},
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = AsyncClient(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile():
    response = await client.create('dummy', files={'test.gguf': 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'})
    assert response['status'] == 'success'


async def test_async_client_create_with_parameters_roundtrip(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'quantize': 'q4_k_m',
      'from': 'mymodel',
      'adapters': {'someadapter.gguf': 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'},
      'template': '[INST] <<SYS>>{{.System}}<</SYS>>\n{{.Prompt}} [/INST]',
      'license': 'this is my license',
      'system': '\nUse\nmultiline\nstrings.\n',
      'parameters': {'stop': ['[INST]', '[/INST]', '<<SYS>>', '<</SYS>>'], 'pi': 3.14159},
      'messages': [{'role': 'user', 'content': 'Hello there!'}, {'role': 'assistant', 'content': 'Hello there yourself!'}],
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = AsyncClient(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile():
    response = await client.create(
      'dummy',
      quantize='q4_k_m',
      from_='mymodel',
      adapters={'someadapter.gguf': 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'},
      template='[INST] <<SYS>>{{.System}}<</SYS>>\n{{.Prompt}} [/INST]',
      license='this is my license',
      system='\nUse\nmultiline\nstrings.\n',
      parameters={'stop': ['[INST]', '[/INST]', '<<SYS>>', '<</SYS>>'], 'pi': 3.14159},
      messages=[{'role': 'user', 'content': 'Hello there!'}, {'role': 'assistant', 'content': 'Hello there yourself!'}],
      stream=False,
    )
    assert response['status'] == 'success'


async def test_async_client_create_from_library(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'from': 'llama2',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = AsyncClient(httpserver.url_for('/'))

  response = await client.create('dummy', from_='llama2')
  assert response['status'] == 'success'


async def test_async_client_create_blob(httpserver: HTTPServer):
  httpserver.expect_ordered_request(re.compile('^/api/blobs/sha256[:-][0-9a-fA-F]{64}$'), method='POST').respond_with_response(Response(status=201))

  client = AsyncClient(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = await client.create_blob(blob.name)
    assert response == 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


async def test_async_client_create_blob_exists(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))

  client = AsyncClient(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = await client.create_blob(blob.name)
    assert response == 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


async def test_async_client_delete(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/delete'), method='DELETE').respond_with_response(Response(status=200))
  client = AsyncClient(httpserver.url_for('/api/delete'))
  response = await client.delete('dummy')
  assert response['status'] == 'success'


async def test_async_client_copy(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/copy'), method='POST').respond_with_response(Response(status=200))
  client = AsyncClient(httpserver.url_for('/api/copy'))
  response = await client.copy('dum', 'dummer')
  assert response['status'] == 'success'


def test_headers():
  client = Client()
  assert client._client.headers['content-type'] == 'application/json'
  assert client._client.headers['accept'] == 'application/json'
  assert client._client.headers['user-agent'].startswith('ollama-python/')

  client = Client(
    headers={
      'X-Custom': 'value',
      'Content-Type': 'text/plain',
    }
  )
  assert client._client.headers['x-custom'] == 'value'
  assert client._client.headers['content-type'] == 'application/json'


def test_copy_tools():
  def func1(x: int) -> str:
    """Simple function 1.
    Args:
        x (integer): A number
    """

  def func2(y: str) -> int:
    """Simple function 2.
    Args:
        y (string): A string
    """

  # Test with list of functions
  tools = list(_copy_tools([func1, func2]))
  assert len(tools) == 2
  assert tools[0].function.name == 'func1'
  assert tools[1].function.name == 'func2'

  # Test with empty input
  assert list(_copy_tools()) == []
  assert list(_copy_tools(None)) == []
  assert list(_copy_tools([])) == []

  # Test with mix of functions and tool dicts
  tool_dict = {
    'type': 'function',
    'function': {
      'name': 'test',
      'description': 'Test function',
      'parameters': {
        'type': 'object',
        'properties': {'x': {'type': 'string', 'description': 'A string', 'enum': ['a', 'b', 'c']}, 'y': {'type': ['integer', 'number'], 'description': 'An integer'}},
        'required': ['x'],
      },
    },
  }

  tools = list(_copy_tools([func1, tool_dict]))
  assert len(tools) == 2
  assert tools[0].function.name == 'func1'
  assert tools[1].function.name == 'test'


def test_tool_validation():
  arbitrary_tool = {'type': 'custom_type', 'function': {'name': 'test'}}
  tools = list(_copy_tools([arbitrary_tool]))
  assert len(tools) == 1
  assert tools[0].type == 'custom_type'
  assert tools[0].function.name == 'test'


def test_client_connection_error():
  client = Client('http://localhost:1234')

  with pytest.raises(ConnectionError, match=CONNECTION_ERROR_MESSAGE):
    client.chat('model', messages=[{'role': 'user', 'content': 'prompt'}])
  with pytest.raises(ConnectionError, match=CONNECTION_ERROR_MESSAGE):
    client.chat('model', messages=[{'role': 'user', 'content': 'prompt'}])
  with pytest.raises(ConnectionError, match=CONNECTION_ERROR_MESSAGE):
    client.generate('model', 'prompt')
  with pytest.raises(ConnectionError, match=CONNECTION_ERROR_MESSAGE):
    client.show('model')


async def test_async_client_connection_error():
  client = AsyncClient('http://localhost:1234')
  with pytest.raises(ConnectionError) as exc_info:
    await client.chat('model', messages=[{'role': 'user', 'content': 'prompt'}])
  assert str(exc_info.value) == 'Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download'
  with pytest.raises(ConnectionError) as exc_info:
    await client.generate('model', 'prompt')
  assert str(exc_info.value) == 'Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download'
  with pytest.raises(ConnectionError) as exc_info:
    await client.show('model')
  assert str(exc_info.value) == 'Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download'


def test_arbitrary_roles_accepted_in_message():
  _ = Message(role='somerandomrole', content="I'm ok with you adding any role message now!")


def _mock_request(*args: Any, **kwargs: Any) -> Response:
  return httpxResponse(status_code=200, content="{'response': 'Hello world!'}")


def test_arbitrary_roles_accepted_in_message_request(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(Client, '_request', _mock_request)

  client = Client()

  client.chat(model='llama3.1', messages=[{'role': 'somerandomrole', 'content': "I'm ok with you adding any role message now!"}, {'role': 'user', 'content': 'Hello world!'}])


async def _mock_request_async(*args: Any, **kwargs: Any) -> Response:
  return httpxResponse(status_code=200, content="{'response': 'Hello world!'}")


async def test_arbitrary_roles_accepted_in_message_request_async(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(AsyncClient, '_request', _mock_request_async)

  client = AsyncClient()

  await client.chat(model='llama3.1', messages=[{'role': 'somerandomrole', 'content': "I'm ok with you adding any role message now!"}, {'role': 'user', 'content': 'Hello world!'}])
