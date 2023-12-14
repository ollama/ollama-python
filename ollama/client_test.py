import pytest
import os
import io
import types
import tempfile
from pathlib import Path
from ollama.client import Client
from pytest_httpserver import HTTPServer, URIPattern
from werkzeug.wrappers import Response
from PIL import Image


class PrefixPattern(URIPattern):
  def __init__(self, prefix: str):
    self.prefix = prefix

  def match(self, uri):
    return uri.startswith(self.prefix)


def test_client_chat(httpserver: HTTPServer):
  httpserver.expect_ordered_request('/api/chat', method='POST', json={
    'model': 'dummy',
    'messages': [{'role': 'user', 'content': 'Why is the sky blue?'}],
    'stream': False,
    'format': '',
    'options': {},
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))
  response = client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
  assert isinstance(response, dict)


def test_client_chat_stream(httpserver: HTTPServer):
  httpserver.expect_ordered_request('/api/chat', method='POST', json={
    'model': 'dummy',
    'messages': [{'role': 'user', 'content': 'Why is the sky blue?'}],
    'stream': True,
    'format': '',
    'options': {},
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))
  response = client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}], stream=True)
  assert isinstance(response, types.GeneratorType)


def test_client_chat_images(httpserver: HTTPServer):
  httpserver.expect_ordered_request('/api/chat', method='POST', json={
    'model': 'dummy',
    'messages': [
      {
        'role': 'user',
        'content': 'Why is the sky blue?',
        'images': ['iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC'],
      },
    ],
    'stream': False,
    'format': '',
    'options': {},
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))

  with io.BytesIO() as b:
    Image.new('RGB', (1, 1)).save(b, 'PNG')
    response = client.chat('dummy', messages=[{'role': 'user', 'content': 'Why is the sky blue?', 'images': [b.getvalue()]}])
    assert isinstance(response, dict)


def test_client_generate(httpserver: HTTPServer):
  httpserver.expect_ordered_request('/api/generate', method='POST', json={
    'model': 'dummy',
    'prompt': 'Why is the sky blue?',
    'system': '',
    'template': '',
    'context': [],
    'stream': False,
    'raw': False,
    'images': [],
    'format': '',
    'options': {},
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))
  response = client.generate('dummy', 'Why is the sky blue?')
  assert isinstance(response, dict)


def test_client_generate_stream(httpserver: HTTPServer):
  httpserver.expect_ordered_request('/api/generate', method='POST', json={
    'model': 'dummy',
    'prompt': 'Why is the sky blue?',
    'system': '',
    'template': '',
    'context': [],
    'stream': True,
    'raw': False,
    'images': [],
    'format': '',
    'options': {},
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))
  response = client.generate('dummy', 'Why is the sky blue?', stream=True)
  assert isinstance(response, types.GeneratorType)


def test_client_generate_images(httpserver: HTTPServer):
  httpserver.expect_ordered_request('/api/generate', method='POST', json={
    'model': 'dummy',
    'prompt': 'Why is the sky blue?',
    'system': '',
    'template': '',
    'context': [],
    'stream': False,
    'raw': False,
    'images': ['iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC'],
    'format': '',
    'options': {},
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as temp:
    Image.new('RGB', (1, 1)).save(temp, 'PNG')
    response = client.generate('dummy', 'Why is the sky blue?', images=[temp.name])
    assert isinstance(response, dict)


def test_client_pull(httpserver: HTTPServer):
  httpserver.expect_ordered_request('/api/pull', method='POST', json={
    'model': 'dummy',
    'insecure': False,
    'stream': False,
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))
  response = client.pull('dummy')
  assert isinstance(response, dict)


def test_client_pull_stream(httpserver: HTTPServer):
  httpserver.expect_ordered_request('/api/pull', method='POST', json={
    'model': 'dummy',
    'insecure': False,
    'stream': True,
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))
  response = client.pull('dummy', stream=True)
  assert isinstance(response, types.GeneratorType)


def test_client_push(httpserver: HTTPServer):
  httpserver.expect_ordered_request('/api/push', method='POST', json={
    'model': 'dummy',
    'insecure': False,
    'stream': False,
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))
  response = client.push('dummy')
  assert isinstance(response, dict)


def test_client_push_stream(httpserver: HTTPServer):
  httpserver.expect_ordered_request('/api/push', method='POST', json={
    'model': 'dummy',
    'insecure': False,
    'stream': True,
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))
  response = client.push('dummy', stream=True)
  assert isinstance(response, types.GeneratorType)


def test_client_create_path(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='HEAD').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request('/api/create', method='POST', json={
    'model': 'dummy',
    'modelfile': 'FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n',
    'stream': False,
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as modelfile:
    with tempfile.NamedTemporaryFile() as blob:
      modelfile.write(f'FROM {blob.name}'.encode('utf-8'))
      modelfile.flush()

      response = client.create('dummy', path=modelfile.name)
      assert isinstance(response, dict)


def test_client_create_path_relative(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='HEAD').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request('/api/create', method='POST', json={
    'model': 'dummy',
    'modelfile': 'FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n',
    'stream': False,
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as modelfile:
    with tempfile.NamedTemporaryFile(dir=Path(modelfile.name).parent) as blob:
      modelfile.write(f'FROM {Path(blob.name).name}'.encode('utf-8'))
      modelfile.flush()

      response = client.create('dummy', path=modelfile.name)
      assert isinstance(response, dict)


@pytest.fixture
def userhomedir():
  with tempfile.TemporaryDirectory() as temp:
    home = os.getenv('HOME', '')
    os.environ['HOME'] = temp
    yield Path(temp)
    os.environ['HOME'] = home


def test_client_create_path_user_home(httpserver: HTTPServer, userhomedir):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='HEAD').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request('/api/create', method='POST', json={
    'model': 'dummy',
    'modelfile': 'FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n',
    'stream': False,
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as modelfile:
    with tempfile.NamedTemporaryFile(dir=userhomedir) as blob:
      modelfile.write(f'FROM ~/{Path(blob.name).name}'.encode('utf-8'))
      modelfile.flush()

      response = client.create('dummy', path=modelfile.name)
      assert isinstance(response, dict)


def test_client_create_modelfile(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='HEAD').respond_with_response(Response(status=200))
  httpserver.expect_ordered_request('/api/create', method='POST', json={
    'model': 'dummy',
    'modelfile': 'FROM @sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n',
    'stream': False,
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = client.create('dummy', modelfile=f'FROM {blob.name}')
    assert isinstance(response, dict)


def test_client_create_from_library(httpserver: HTTPServer):
  httpserver.expect_ordered_request('/api/create', method='POST', json={
    'model': 'dummy',
    'modelfile': 'FROM llama2\n',
    'stream': False,
  }).respond_with_json({})

  client = Client(httpserver.url_for('/'))

  response = client.create('dummy', modelfile='FROM llama2')
  assert isinstance(response, dict)


def test_client_create_blob(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='HEAD').respond_with_response(Response(status=404))
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='PUT').respond_with_response(Response(status=201))

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = client.create_blob(blob.name)
    assert response == 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


def test_client_create_blob_exists(httpserver: HTTPServer):
  httpserver.expect_ordered_request(PrefixPattern('/api/blobs/'), method='HEAD').respond_with_response(Response(status=200))

  client = Client(httpserver.url_for('/'))

  with tempfile.NamedTemporaryFile() as blob:
    response = client.create_blob(blob.name)
    assert response == 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
