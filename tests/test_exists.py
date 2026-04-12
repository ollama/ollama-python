import json

import pytest
from pytest_httpserver import HTTPServer

from ollama._client import AsyncClient, Client


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
  return 'asyncio'


def test_client_exists_true(httpserver: HTTPServer):
  """exists() returns True when model is present."""
  httpserver.expect_ordered_request(
    '/api/tags',
    method='GET',
  ).respond_with_json({
    'models': [
      {'name': 'llama3.1:8b', 'model': 'llama3.1:8b', 'size': 4661224676},
      {'name': 'qwen2.5:latest', 'model': 'qwen2.5:latest', 'size': 4430121000},
    ]
  })

  client = Client(host=f'http://{httpserver.host}:{httpserver.port}')
  assert client.exists('llama3.1:8b') is True


def test_client_exists_false(httpserver: HTTPServer):
  """exists() returns False when model is not present."""
  httpserver.expect_ordered_request(
    '/api/tags',
    method='GET',
  ).respond_with_json({
    'models': [
      {'name': 'llama3.1:8b', 'model': 'llama3.1:8b', 'size': 4661224676},
    ]
  })

  client = Client(host=f'http://{httpserver.host}:{httpserver.port}')
  assert client.exists('gemma2:2b') is False


def test_client_exists_empty_list(httpserver: HTTPServer):
  """exists() returns False when no models are available."""
  httpserver.expect_ordered_request(
    '/api/tags',
    method='GET',
  ).respond_with_json({'models': []})

  client = Client(host=f'http://{httpserver.host}:{httpserver.port}')
  assert client.exists('llama3.1:8b') is False


async def test_async_client_exists_true(httpserver: HTTPServer):
  """Async exists() returns True when model is present."""
  httpserver.expect_ordered_request(
    '/api/tags',
    method='GET',
  ).respond_with_json({
    'models': [
      {'name': 'llama3.1:8b', 'model': 'llama3.1:8b', 'size': 4661224676},
    ]
  })

  client = AsyncClient(host=f'http://{httpserver.host}:{httpserver.port}')
  assert await client.exists('llama3.1:8b') is True


async def test_async_client_exists_false(httpserver: HTTPServer):
  """Async exists() returns False when model is not present."""
  httpserver.expect_ordered_request(
    '/api/tags',
    method='GET',
  ).respond_with_json({'models': []})

  client = AsyncClient(host=f'http://{httpserver.host}:{httpserver.port}')
  assert await client.exists('nonexistent') is False
