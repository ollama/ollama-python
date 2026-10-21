import asyncio
import inspect
import json
from functools import partial, update_wrapper, wraps
from typing import Optional

import httpx
import pytest

from ollama import AsyncClient, Client
from ollama._utils import convert_function_to_tool


def scoped_lookup(tenant: str, query: str) -> str:
  """Look up a record.

  Args:
    tenant: The application tenant.
    query: The record to retrieve.
  """
  return tenant + ':' + query


def make_tool(case):
  tool = partial(scoped_lookup, 'bound-tenant-fixture')
  name, description = 'scoped_lookup', 'Look up a record.'
  parameters = ['query']
  if case == 'custom_name':
    tool.__name__ = name = 'tenant_lookup'
  elif case == 'custom_doc':
    tool.__doc__ = 'Search this tenant.\n\nArgs:\n  query: The record to retrieve.'
    description = 'Search this tenant.'
  elif case == 'wrapped':
    update_wrapper(tool, scoped_lookup)
  elif case == 'nested_wrapped':
    update_wrapper(tool, scoped_lookup)
    tool = partial(tool, 'bound-query-fixture')
    parameters = []
  elif case == 'decorated_function':

    @wraps(scoped_lookup)
    def decorated(*args, **kwargs):
      return scoped_lookup(*args, **kwargs)

    tool = partial(decorated, 'bound-tenant-fixture')
  elif case == 'explicit_signature':
    tool.__signature__ = inspect.Signature([inspect.Parameter('query', inspect.Parameter.KEYWORD_ONLY, annotation=str)])
  return tool, name, description, parameters


def chat_with_tool(tool, asynchronous):
  requests = []

  def handle(request):
    requests.append(json.loads(request.content))
    return httpx.Response(200, json={'model': 'fixture', 'message': {'role': 'assistant', 'content': 'ok'}, 'done': True})

  async def async_chat():
    async with AsyncClient(host='http://offline.invalid', transport=httpx.MockTransport(handle), trust_env=False) as client:
      response = await client.chat('fixture', tools=[tool])
      assert response.message.content == 'ok'

  if asynchronous:
    asyncio.run(async_chat())
  else:
    with Client(host='http://offline.invalid', transport=httpx.MockTransport(handle), trust_env=False) as client:
      response = client.chat('fixture', tools=[tool])
      assert response.message.content == 'ok'

  assert len(requests) == 1
  return requests[0]['tools'][0]['function']


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('case', ['plain', 'custom_name', 'custom_doc', 'wrapped', 'nested_wrapped', 'decorated_function', 'explicit_signature'])
def test_chat_partial_tool(case, asynchronous):
  tool, name, description, parameters = make_tool(case)
  before = vars(tool).copy()
  function = chat_with_tool(tool, asynchronous)

  assert function['name'] == name
  assert function['description'].strip() == description
  assert set(function['parameters']['properties']) == set(parameters)
  assert function['parameters'].get('required', []) == parameters
  if parameters:
    assert function['parameters']['properties']['query'] == {'type': 'string', 'description': 'The record to retrieve.'}
  assert 'bound-tenant-fixture' not in json.dumps(function)
  assert 'bound-query-fixture' not in json.dumps(function)
  assert vars(tool) == before


@pytest.mark.parametrize('asynchronous', [False, True])
def test_partial_keyword_bindings_keep_existing_function_schema(asynchronous):
  def lookup(tenant: str, query: str, limit: int = 5, label: Optional[str] = None) -> str:
    """Look up a record.

    Args:
      tenant: The application tenant.
      query: The record to retrieve.
      limit: The result limit.
      label: An optional label.
    """
    return tenant + ':' + query

  def ordinary(query: str, *, limit: int = 5, label: Optional[str] = None) -> str:
    """Look up a record.

    Args:
      query: The record to retrieve.
      limit: The result limit.
      label: An optional label.
    """
    return query

  tool = partial(lookup, 'bound-tenant-fixture', limit=5, label=None)
  function = chat_with_tool(tool, asynchronous)
  ordinary_schema = chat_with_tool(ordinary, asynchronous)
  assert function['parameters'] == ordinary_schema['parameters']
  assert set(function['parameters']['properties']) == {'query', 'limit', 'label'}
  assert 'tenant' not in function['parameters']['properties']
  assert 'label' not in function['parameters'].get('required', [])
  assert tool(query='query-fixture', limit=10) == 'bound-tenant-fixture:query-fixture'


def test_plain_decorated_functions_keep_their_signature():
  @wraps(scoped_lookup)
  def decorated(*args, **kwargs):
    return scoped_lookup(*args, **kwargs)

  assert convert_function_to_tool(decorated) == convert_function_to_tool(scoped_lookup)
