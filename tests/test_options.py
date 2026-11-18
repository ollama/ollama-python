import warnings

import pytest
from pytest_httpserver import HTTPServer

from ollama._client import Client
from ollama._types import ChatRequest, CreateRequest, Options


def test_options_drop_unsupported_keywords():
  with pytest.warns(FutureWarning, match='typical_p'):
    options = Options(typical_p=0.5, temperature=0.1)
  assert options.model_dump(exclude_none=True) == {'temperature': 0.1}


def test_options_drop_unsupported_item_assignment():
  options = Options(temperature=0.1)
  with pytest.warns(FutureWarning, match='mirostat'):
    options['mirostat'] = 2
  assert options.model_dump(exclude_none=True) == {'temperature': 0.1}


def test_options_keep_unknown_keys():
  with warnings.catch_warnings():
    warnings.simplefilter('error')
    options = Options(future_option=1, min_p=0.05)
  assert options.model_dump(exclude_none=True) == {'future_option': 1, 'min_p': 0.05}


def test_request_drops_unsupported_options_mapping():
  with pytest.warns(FutureWarning, match='typical_p'):
    request = ChatRequest(model='dummy', messages=[], options={'typical_p': 0.5, 'num_ctx': 8})
  assert request.model_dump(exclude_none=True)['options'] == {'num_ctx': 8}


def test_create_request_drops_unsupported_parameters():
  with pytest.warns(FutureWarning, match='penalize_newline'):
    request = CreateRequest(model='dummy', parameters={'penalize_newline': True, 'pi': 3.14})
  assert request.model_dump(exclude_none=True)['parameters'] == {'pi': 3.14}


def test_client_chat_drops_unsupported_options(httpserver: HTTPServer):
  httpserver.expect_ordered_request(
    '/api/chat',
    method='POST',
    json={
      'model': 'dummy',
      'messages': [{'role': 'user', 'content': 'Hi'}],
      'tools': [],
      'stream': False,
      'options': {'temperature': 0.0},
    },
  ).respond_with_json({'model': 'dummy', 'message': {'role': 'assistant', 'content': 'Hello'}})

  client = Client(httpserver.url_for('/'))
  with pytest.warns(FutureWarning, match='typical_p'):
    response = client.chat('dummy', messages=[{'role': 'user', 'content': 'Hi'}], options={'typical_p': 0.5, 'temperature': 0.0})
  assert response['message']['content'] == 'Hello'
