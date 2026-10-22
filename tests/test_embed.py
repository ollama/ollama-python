import pytest
from pydantic import ValidationError

from ollama import AsyncClient, Client, EmbedResponse, Options, ResponseError

pytestmark = pytest.mark.anyio

# A 1x1 PNG and a one-sample PCM WAV. Responses below are HTTP fixtures, not inference.
PNG_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC'
WAV_BASE64 = 'UklGRiYAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQIAAAAAAA=='


@pytest.fixture
def anyio_backend():
  return 'asyncio'


async def embed(httpserver, async_mode, **kwargs):
  if async_mode:
    async with AsyncClient(httpserver.url_for('/')) as client:
      return await client.embed(**kwargs)
  with Client(httpserver.url_for('/')) as client:
    return client.embed(**kwargs)


@pytest.mark.parametrize('async_mode', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize(
  'input',
  [
    {'text': 'A tiny image 🖼️', 'image': PNG_BASE64},
    {'image': PNG_BASE64},
    {'text': 'A short sound', 'audio': WAV_BASE64},
    {'audio': WAV_BASE64},
    ['plain text', {'text': '图片', 'image': PNG_BASE64}, {'audio': WAV_BASE64}],
    ({'image': PNG_BASE64}, 'another item'),
  ],
  ids=['text-image', 'image', 'text-audio', 'audio', 'mixed-list', 'mixed-tuple'],
)
async def test_embed_structured_inputs(httpserver, async_mode, input):
  wire_input = list(input) if isinstance(input, tuple) else input
  count = len(wire_input) if isinstance(wire_input, list) else 1
  embeddings = [[float(i), i + 0.5] for i in range(count)]
  httpserver.expect_ordered_request('/api/embed', method='POST', json={'model': 'dummy', 'input': wire_input}).respond_with_json({'model': 'dummy', 'embeddings': embeddings, 'prompt_eval_count': 3})

  response = await embed(httpserver, async_mode, model='dummy', input=input)

  assert isinstance(response, EmbedResponse)
  assert response.embeddings == embeddings
  assert response['prompt_eval_count'] == 3
  httpserver.check_assertions()


@pytest.mark.parametrize('async_mode', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('input', ['hello', ['one', 'two'], ('one', 'two'), '', []], ids=['scalar', 'list', 'tuple', 'empty-string', 'empty-list'])
async def test_embed_text_inputs(httpserver, async_mode, input):
  wire_input = list(input) if isinstance(input, tuple) else input
  httpserver.expect_ordered_request('/api/embed', method='POST', json={'model': 'dummy', 'input': wire_input}).respond_with_json({'embeddings': [[0.25]]})

  response = await embed(httpserver, async_mode, model='dummy', input=input)

  assert response.embeddings == [[0.25]]
  httpserver.check_assertions()


@pytest.mark.parametrize('async_mode', [False, True], ids=['sync', 'async'])
async def test_embed_default_input(httpserver, async_mode):
  httpserver.expect_ordered_request('/api/embed', method='POST', json={'model': 'dummy', 'input': ''}).respond_with_json({'embeddings': []})

  response = await embed(httpserver, async_mode, model='dummy')

  assert response.embeddings == []
  httpserver.check_assertions()


@pytest.mark.parametrize('async_mode', [False, True], ids=['sync', 'async'])
async def test_embed_missing_model_still_rejected(httpserver, async_mode):
  with pytest.raises(ValidationError, match='model'):
    await embed(httpserver, async_mode)
  assert not httpserver.log


@pytest.mark.parametrize('async_mode', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('options_model', [False, True], ids=['mapping', 'Options'])
async def test_embed_structured_options(httpserver, async_mode, options_model):
  input = {'image': PNG_BASE64}
  options = {'num_ctx': 128}
  httpserver.expect_ordered_request(
    '/api/embed',
    method='POST',
    json={'model': 'dummy', 'input': input, 'truncate': False, 'options': options, 'keep_alive': '5m', 'dimensions': 2},
  ).respond_with_json({'embeddings': [[0.25, 0.5]], 'total_duration': 12})

  response = await embed(httpserver, async_mode, model='dummy', input=input, truncate=False, options=Options(**options) if options_model else options, keep_alive='5m', dimensions=2)

  assert response.embeddings == [[0.25, 0.5]]
  assert response.total_duration == 12
  httpserver.check_assertions()


@pytest.mark.parametrize('async_mode', [False, True], ids=['sync', 'async'])
async def test_embed_structured_server_error(httpserver, async_mode):
  # Media decoding and model support remain server concerns, not SDK file loading.
  input = {'image': 'invalid-base64'}
  httpserver.expect_ordered_request('/api/embed', method='POST', json={'model': 'dummy', 'input': input}).respond_with_json({'error': 'input.image: invalid base64'}, status=400)

  with pytest.raises(ResponseError, match=r'input\.image: invalid base64') as error:
    await embed(httpserver, async_mode, model='dummy', input=input)

  assert error.value.status_code == 400
  httpserver.check_assertions()
