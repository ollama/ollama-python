import tempfile
from base64 import b64encode
from pathlib import Path

import pytest

from ollama._types import CreateRequest, Image


def test_image_serialization_bytes():
  image_bytes = b'test image bytes'
  encoded_string = b64encode(image_bytes).decode()
  img = Image(value=image_bytes)
  assert img.model_dump() == encoded_string


def test_image_serialization_base64_string():
  b64_str = 'dGVzdCBiYXNlNjQgc3RyaW5n'
  img = Image(value=b64_str)
  assert img.model_dump() == b64_str  # Should return as-is if valid base64


def test_image_serialization_long_base64_string():
  b64_str = 'dGVzdCBiYXNlNjQgc3RyaW5n' * 1000
  img = Image(value=b64_str)
  assert img.model_dump() == b64_str  # Should return as-is if valid base64


def test_image_serialization_plain_string():
  img = Image(value='not a path or base64')
  assert img.model_dump() == 'not a path or base64'  # Should return as-is


def test_image_serialization_path():
  with tempfile.NamedTemporaryFile() as temp_file:
    temp_file.write(b'test file content')
    temp_file.flush()
    img = Image(value=Path(temp_file.name))
    assert img.model_dump() == b64encode(b'test file content').decode()


def test_image_serialization_string_path():
  with tempfile.NamedTemporaryFile() as temp_file:
    temp_file.write(b'test file content')
    temp_file.flush()
    img = Image(value=temp_file.name)
    assert img.model_dump() == b64encode(b'test file content').decode()

  with pytest.raises(ValueError):
    img = Image(value='some_path/that/does/not/exist.png')
    img.model_dump()

  with pytest.raises(ValueError):
    img = Image(value='not an image')
    img.model_dump()


def test_create_request_serialization():
  request = CreateRequest(model='test-model', from_='base-model', quantize='q4_0', files={'file1': 'content1'}, adapters={'adapter1': 'content1'}, template='test template', license='MIT', system='test system', parameters={'param1': 'value1'})

  serialized = request.model_dump()
  assert serialized['from'] == 'base-model'
  assert 'from_' not in serialized
  assert serialized['quantize'] == 'q4_0'
  assert serialized['files'] == {'file1': 'content1'}
  assert serialized['adapters'] == {'adapter1': 'content1'}
  assert serialized['template'] == 'test template'
  assert serialized['license'] == 'MIT'
  assert serialized['system'] == 'test system'
  assert serialized['parameters'] == {'param1': 'value1'}


def test_create_request_serialization_exclude_none_true():
  request = CreateRequest(model='test-model', from_=None, quantize=None)
  serialized = request.model_dump(exclude_none=True)
  assert serialized == {'model': 'test-model'}
  assert 'from' not in serialized
  assert 'from_' not in serialized
  assert 'quantize' not in serialized


def test_create_request_serialization_exclude_none_false():
  request = CreateRequest(model='test-model', from_=None, quantize=None)
  serialized = request.model_dump(exclude_none=False)
  assert 'from' in serialized
  assert 'quantize' in serialized
  assert 'adapters' in serialized
  assert 'from_' not in serialized


def test_create_request_serialization_license_list():
  request = CreateRequest(model='test-model', license=['MIT', 'Apache-2.0'])
  serialized = request.model_dump()
  assert serialized['license'] == ['MIT', 'Apache-2.0']
