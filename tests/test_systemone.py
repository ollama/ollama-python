import inspect
import json

import httpx
import pytest

import ollama


@pytest.fixture
def anyio_backend():
  return 'asyncio'


@pytest.mark.anyio
@pytest.mark.parametrize('variant', ['sync', 'async', 'module'])
async def test_systemone_wire_and_answers(variant, monkeypatch):
  questions = {
    'team': {'type': 'choice', 'instructions': {'task': 'Route ticket'}, 'criteria': {'technical': None, 'billing': 'Payments'}},
    'refund': ollama.SystemOneNoulQuestion(instructions=['Refund requested?']),
    'urgency': {'type': 'score', 'instructions': 'Urgency?', 'criteria': ['Routine', 'Urgent']},
  }
  expected = {
    'model': 'nimble',
    'state': {'ticket': ['Duplicate charge', None], 'paid': True},
    'questions': {
      'team': {'type': 'choice', 'instructions': {'task': 'Route ticket'}, 'criteria': {'technical': None, 'billing': 'Payments'}},
      'refund': {'type': 'noul', 'instructions': ['Refund requested?']},
      'urgency': {'type': 'score', 'instructions': 'Urgency?', 'criteria': ['Routine', 'Urgent']},
    },
  }
  # Exercise both omission and numeric zero without duplicating the client matrix.
  if variant == 'sync':
    expected['keep_alive'] = 0
  elif variant == 'async':
    expected['keep_alive'] = '5m'
  result = {
    'model': 'nimble',
    'answers': {
      'team': {'type': 'choice', 'choice': 'billing', 'probabilities': {'technical': 0.2, 'billing': 0.8}, 'confidence': 0.3},
      'refund': {'type': 'noul', 'noul': 0.9},
      'urgency': {'type': 'score', 'score': 0.25, 'legend': {'0': 'Routine', '1': 'Urgent'}, 'probabilities': {'0': 0.75, '1': 0.25}, 'confidence': 0.2},
    },
    'usage': {'input_tokens': 123, 'output_tokens': 4},
  }

  def handle(request):
    assert request.method == 'POST'
    assert str(request.url) == 'http://sdk.test:11474/v1/systemone'
    assert request.headers['authorization'] == 'Bearer synthetic'
    body = json.loads(request.content)
    assert body == expected
    assert list(body['questions']) == ['team', 'refund', 'urgency']
    assert list(body['questions']['team']['criteria']) == ['technical', 'billing']
    return httpx.Response(200, json=result)

  transport = httpx.MockTransport(handle)
  if variant == 'module':
    monkeypatch.setattr(ollama._client, '_client', httpx.Client(base_url='http://sdk.test:11474', headers={'authorization': 'Bearer synthetic'}, transport=transport))
    method = ollama.systemone
  else:
    client = (ollama.AsyncClient if variant == 'async' else ollama.Client)(host='http://sdk.test:11474', headers={'authorization': 'Bearer synthetic'}, transport=transport)
    method = client.systemone
  kwargs = {key: expected[key] for key in ('model', 'state', 'keep_alive') if key in expected}
  response = method(**kwargs, questions=questions)
  if inspect.isawaitable(response):
    response = await response
  assert response.model_dump() == result
  assert isinstance(response.answers['team'], ollama.SystemOneChoiceAnswer)
  assert isinstance(response.answers['refund'], ollama.SystemOneNoulAnswer)
  assert isinstance(response.answers['urgency'], ollama.SystemOneScoreAnswer)
  assert response['answers']['urgency']['score'] == 0.25


@pytest.mark.anyio
@pytest.mark.parametrize('client_type', [ollama.Client, ollama.AsyncClient])
@pytest.mark.parametrize('status,message', [(400, 'state must not be empty'), (404, "model 'missing' not found"), (413, 'request body must not exceed 64 KiB'), (503, 'server busy')])
async def test_systemone_http_errors(client_type, status, message):
  def handle(request):
    return httpx.Response(status, json={'error': message})

  client = client_type(transport=httpx.MockTransport(handle))
  with pytest.raises(ollama.ResponseError) as error:
    response = client.systemone('nimble', 'ticket', {'refund': {'type': 'noul', 'instructions': 'Refund?'}})
    if inspect.isawaitable(response):
      await response
  assert error.value.status_code == status
  assert error.value.error == message
