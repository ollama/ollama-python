from base64 import b64decode, b64encode
from datetime import datetime

from ollama._types import GenerateRequest, ChatRequest, Message, Image, Tool, Options, EmbedRequest, EmbedResponse, PullRequest, PushRequest, ShowRequest, ShowResponse, ProcessResponse, ListResponse, RequestError, ResponseError


def test_generate_request_serialization():
  req = GenerateRequest(model='test-model', prompt='test prompt', system='test system', template='test template', context=[1, 2, 3], options={'temperature': 0.7})

  data = req.model_dump(exclude_none=True)
  assert data == {'model': 'test-model', 'prompt': 'test prompt', 'system': 'test system', 'template': 'test template', 'context': [1, 2, 3], 'options': {'temperature': 0.7}}


def test_chat_request_serialization():
  req = ChatRequest(model='test-model', messages=[Message(role='user', content='Hello'), Message(role='assistant', content='Hi there')])

  data = req.model_dump(exclude_none=True)
  assert data == {'model': 'test-model', 'messages': [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there'}]}


def test_image_serialization():
  # Test bytes serialization
  image_bytes = b'test image bytes'
  img = Image(value=image_bytes)
  assert img.model_dump() == b64encode(image_bytes).decode()

  # Test base64 string serialization
  b64_str = 'dGVzdCBiYXNlNjQgc3RyaW5n'
  img = Image(value=b64_str)
  print(img.model_dump())
  assert img.model_dump() == b64decode(b64_str).decode()


def test_tool_serialization():
  tool = Tool(type='function', function=Tool.Function(name='test_function', description='A test function', parameters=Tool.Function.Parameters(type='object', required=['param1'], properties={'param1': {'type': 'string'}})))

  data = tool.model_dump(exclude_none=True)
  assert data == {'type': 'function', 'function': {'name': 'test_function', 'description': 'A test function', 'parameters': {'type': 'object', 'required': ['param1'], 'properties': {'param1': {'type': 'string'}}}}}


def test_options_serialization():
  options = Options(num_ctx=4096, num_gpu=1, temperature=0.7, top_p=0.9, stop=['.', '?', '!'])

  data = options.model_dump(exclude_none=True)
  assert data == {'num_ctx': 4096, 'num_gpu': 1, 'temperature': 0.7, 'top_p': 0.9, 'stop': ['.', '?', '!']}


def test_embed_request_response_serialization():
  # Test EmbedRequest
  req = EmbedRequest(model='test-model', input='test input', truncate=True)

  data = req.model_dump(exclude_none=True)
  assert data == {'model': 'test-model', 'input': 'test input', 'truncate': True}

  # Test EmbedResponse
  resp = EmbedResponse(embeddings=[[0.1, 0.2, 0.3]], total_duration=1000)

  data = resp.model_dump(exclude_none=True)
  assert data == {'embeddings': [[0.1, 0.2, 0.3]], 'total_duration': 1000}


def test_pull_push_request_serialization():
  # Test PullRequest
  pull_req = PullRequest(model='test-model', insecure=True, stream=True)

  data = pull_req.model_dump(exclude_none=True)
  assert data == {'model': 'test-model', 'insecure': True, 'stream': True}

  # Test PushRequest
  push_req = PushRequest(model='test-model', insecure=True, stream=True)

  data = push_req.model_dump(exclude_none=True)
  assert data == {'model': 'test-model', 'insecure': True, 'stream': True}


def test_show_request_response_serialization():
  # Test ShowRequest
  req = ShowRequest(model='test-model')

  data = req.model_dump(exclude_none=True)
  assert data == {'model': 'test-model'}

  # Test ShowResponse
  resp = ShowResponse(modified_at=datetime(2024, 1, 1), template='test template', modelfile='test modelfile', license='MIT', details={'format': 'gguf', 'family': 'llama'}, model_info={'version': '1.0'}, parameters='test params')

  data = resp.model_dump(exclude_none=True)
  assert data == {'modified_at': datetime(2024, 1, 1), 'template': 'test template', 'modelfile': 'test modelfile', 'license': 'MIT', 'details': {'format': 'gguf', 'family': 'llama'}, 'modelinfo': {'version': '1.0'}, 'parameters': 'test params'}


def test_process_response_serialization():
  resp = ProcessResponse(models=[ProcessResponse.Model(model='test-model', name='test', digest='sha256:abc', size=1000, size_vram=500)])

  data = resp.model_dump(exclude_none=True)
  assert data == {'models': [{'model': 'test-model', 'name': 'test', 'digest': 'sha256:abc', 'size': 1000, 'size_vram': 500}]}


def test_list_response_serialization():
  resp = ListResponse(models=[ListResponse.Model(modified_at=datetime(2024, 1, 1), digest='sha256:abc', size=1000, details={'format': 'gguf'})])

  data = resp.model_dump(exclude_none=True)
  assert data == {'models': [{'modified_at': datetime(2024, 1, 1), 'digest': 'sha256:abc', 'size': 1000, 'details': {'format': 'gguf'}}]}


def test_error_classes():
  # Test RequestError
  error = RequestError('test error')
  assert error.error == 'test error'
  assert str(error) == 'test error'

  # Test ResponseError
  error = ResponseError('test error', 404)
  assert error.error == 'test error'
  assert error.status_code == 404
  assert str(error) == 'test error'

  # Test ResponseError with JSON error
  error = ResponseError('{"error": "json error"}', 500)
  assert error.error == 'json error'
  assert error.status_code == 500
