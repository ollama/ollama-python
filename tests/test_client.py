import os
import io
import json
import re
from pydantic import ValidationError, BaseModel
import pytest
import tempfile
from pathlib import Path
from pytest_httpserver import HTTPServer, URIPattern
from werkzeug.wrappers import Request, Response
from PIL import Image

from ollama._client import Client, AsyncClient, _copy_tools


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


def test_client_chat_images(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/chat',
    method='POST',
    json={
      'model': 'dummy',
      'messages': [
        {
          'role': 'user',
          'content': 'Why is the sky blue?',
          'images': ['iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC'],
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

  with io.BytesIO() as b:
    Image.new('RGB', (1, 1)).save(b, 'PNG')
    response = client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?', 'images': [b.getvalue()]}])
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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
      'images': ['iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC'],
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'response': 'Because it is.',
    }
  )

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as temp:
    Image.new('RGB', (1, 1)).save(temp, 'PNG')
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


def test_client_create_path(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'modelfile': 'FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as modelfile:
    with tempfile.NamedTemporaryFile() as blob:
      modelfile.write(f'FROM {blob.name}'.encode('utf-8'))
      modelfile.flush()

      response = client.create('dummy', path=modelfile.name)
      assert response['status'] == 'success'


def test_client_create_path_relative(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'modelfile': 'FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as modelfile:
    with tempfile.NamedTemporaryFile(dir=Path(modelfile.name).parent) as blob:
      modelfile.write(f'FROM {Path(blob.name).name}'.encode('utf-8'))
      modelfile.flush()

      response = client.create('dummy', path=modelfile.name)
      assert response['status'] == 'success'


@pytest.fixture
def userhomedir():
  with tempfile.TemporaryDirectory() as temp:
    home = os.getenv('HOME', '')
    os.environ['HOME'] = temp
    yield Path(temp)
    os.environ['HOME'] = home


def test_client_create_path_user_home(httpserver: HTTPServer, userhomedir):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'modelfile': 'FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as modelfile:
    with tempfile.NamedTemporaryFile(dir=userhomedir) as blob:
      modelfile.write(f'FROM ~/{Path(blob.name).name}'.encode('utf-8'))
      modelfile.flush()

      response = client.create('dummy', path=modelfile.name)
      assert response['status'] == 'success'


def test_client_create_modelfile(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'modelfile': 'FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = client.create('dummy', modelfile=f'FROM {blob.name}')
    assert response['status'] == 'success'


def test_client_create_modelfile_roundtrip(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'modelfile': '''FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
TEMPLATE """[INST] <<SYS>>{{.System}}<</SYS>>
{{.Prompt}} [/INST]"""
SYSTEM """
Use
multiline
strings.
"""
PARAMETER stop [INST]
PARAMETER stop [/INST]
PARAMETER stop <<SYS>>
PARAMETER stop <</SYS>>''',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = client.create(
      'dummy',
      modelfile='\n'.join(
        [
          f'FROM {blob.name}',
          'TEMPLATE """[INST] <<SYS>>{{.System}}<</SYS>>',
          '{{.Prompt}} [/INST]"""',
          'SYSTEM """',
          'Use',
          'multiline',
          'strings.',
          '"""',
          'PARAMETER stop [INST]',
          'PARAMETER stop [/INST]',
          'PARAMETER stop <<SYS>>',
          'PARAMETER stop <</SYS>>',
        ]
      ),
    )
    assert response['status'] == 'success'


def test_client_create_from_library(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'modelfile': 'FROM llama2',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = Client(httpserver.url_for('/'))

  response = client.create('dummy', modelfile='FROM llama2')
  assert response['status'] == 'success'


def test_client_create_blob(httpserver: HTTPServer):
  httpserver.expect_ordered_request(re.compile('^/api/blobs/sha256[:-][0-9a-fA-F]{64}$'), method='POST').respond_with_response(Response(status=201))

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = client._create_blob(blob.name)
    assert response == 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


def test_client_create_blob_exists(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = client._create_blob(blob.name)
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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
          'images': ['iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC'],
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

  with io.BytesIO() as b:
    Image.new('RGB', (1, 1)).save(b, 'PNG')
    response = await client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?', 'images': [b.getvalue()]}])
    assert response['model'] == 'dummy'
    assert response['message']['role'] == 'assistant'
    assert response['message']['content'] == "I don't know."


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_async_client_generate_images(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/generate',
    method='POST',
    json={
      'model': 'dummy',
      'prompt': 'Why is the sky blue?',
      'stream': False,
      'images': ['iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC'],
    },
  ).respond_with_json(
    {
      'model': 'dummy',
      'response': 'Because it is.',
    }
  )

  client = AsyncClient(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as temp:
    Image.new('RGB', (1, 1)).save(temp, 'PNG')
    response = await client.generate('dummy', 'Why is the sky blue?', images=[temp.name])
    assert response['model'] == 'dummy'
    assert response['response'] == 'Because it is.'


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_async_client_create_path(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'modelfile': 'FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = AsyncClient(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as modelfile:
    with tempfile.NamedTemporaryFile() as blob:
      modelfile.write(f'FROM {blob.name}'.encode('utf-8'))
      modelfile.flush()

      response = await client.create('dummy', path=modelfile.name)
      assert response['status'] == 'success'


@pytest.mark.asyncio
async def test_async_client_create_path_relative(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'modelfile': 'FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = AsyncClient(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as modelfile:
    with tempfile.NamedTemporaryFile(dir=Path(modelfile.name).parent) as blob:
      modelfile.write(f'FROM {Path(blob.name).name}'.encode('utf-8'))
      modelfile.flush()

      response = await client.create('dummy', path=modelfile.name)
      assert response['status'] == 'success'


@pytest.mark.asyncio
async def test_async_client_create_path_user_home(httpserver: HTTPServer, userhomedir):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'modelfile': 'FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = AsyncClient(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as modelfile:
    with tempfile.NamedTemporaryFile(dir=userhomedir) as blob:
      modelfile.write(f'FROM ~/{Path(blob.name).name}'.encode('utf-8'))
      modelfile.flush()

      response = await client.create('dummy', path=modelfile.name)
      assert response['status'] == 'success'


@pytest.mark.asyncio
async def test_async_client_create_modelfile(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'modelfile': 'FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = AsyncClient(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = await client.create('dummy', modelfile=f'FROM {blob.name}')
    assert response['status'] == 'success'


@pytest.mark.asyncio
async def test_async_client_create_modelfile_roundtrip(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'modelfile': '''FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
TEMPLATE """[INST] <<SYS>>{{.System}}<</SYS>>
{{.Prompt}} [/INST]"""
SYSTEM """
Use
multiline
strings.
"""
PARAMETER stop [INST]
PARAMETER stop [/INST]
PARAMETER stop <<SYS>>
PARAMETER stop <</SYS>>''',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = AsyncClient(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = await client.create(
      'dummy',
      modelfile='\n'.join(
        [
          f'FROM {blob.name}',
          'TEMPLATE """[INST] <<SYS>>{{.System}}<</SYS>>',
          '{{.Prompt}} [/INST]"""',
          'SYSTEM """',
          'Use',
          'multiline',
          'strings.',
          '"""',
          'PARAMETER stop [INST]',
          'PARAMETER stop [/INST]',
          'PARAMETER stop <<SYS>>',
          'PARAMETER stop <</SYS>>',
        ]
      ),
    )
    assert response['status'] == 'success'


@pytest.mark.asyncio
async def test_async_client_create_from_library(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/create',
    method='POST',
    json={
      'model': 'dummy',
      'modelfile': 'FROM llama2',
      'stream': False,
    },
  ).respond_with_json({'status': 'success'})

  client = AsyncClient(httpserver.url_for('/'))

  response = await client.create('dummy', modelfile='FROM llama2')
  assert response['status'] == 'success'


@pytest.mark.asyncio
async def test_async_client_create_blob(httpserver: HTTPServer):
  httpserver.expect_ordered_request(re.compile('^/api/blobs/sha256[:-][0-9a-fA-F]{64}$'), method='POST').respond_with_response(Response(status=201))

  client = AsyncClient(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = await client._create_blob(blob.name)
    assert response == 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


@pytest.mark.asyncio
async def test_async_client_create_blob_exists(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='POST').respond_with_response(Response(status=200))

  client = AsyncClient(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = await client._create_blob(blob.name)
    assert response == 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


@pytest.mark.asyncio
async def test_async_client_delete(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/delete'), method='DELETE').respond_with_response(Response(status=200))
  client = AsyncClient(httpserver.url_for('/api/delete'))
  response = await client.delete('dummy')
  assert response['status'] == 'success'


@pytest.mark.asyncio
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
    pass

  def func2(y: str) -> int:
    """Simple function 2.
    Args:
        y (string): A string
    """
    pass

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
        'properties': {'x': {'type': 'string', 'description': 'A string'}},
        'required': ['x'],
      },
    },
  }

  tools = list(_copy_tools([func1, tool_dict]))
  assert len(tools) == 2
  assert tools[0].function.name == 'func1'
  assert tools[1].function.name == 'test'


def test_tool_validation():
  # Raises ValidationError when used as it is a generator
  with pytest.raises(ValidationError):
    invalid_tool = {'type': 'invalid_type', 'function': {'name': 'test'}}
    list(_copy_tools([invalid_tool]))
