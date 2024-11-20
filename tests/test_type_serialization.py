from base64 import b64encode
from pathlib import Path
from ollama._types import Image
import tempfile


def test_image_serialization_bytes():
  image_bytes = b'test image bytes'
  encoded_string = b64encode(image_bytes).decode()
  img = Image(value=image_bytes)
  assert img.model_dump() == encoded_string


def test_image_serialization_base64_string():
  b64_str = 'dGVzdCBiYXNlNjQgc3RyaW5n'
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
