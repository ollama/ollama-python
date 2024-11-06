from base64 import b64decode, b64encode

from ollama._types import Image


def test_image_serialization():
  # Test bytes serialization
  image_bytes = b'test image bytes'
  img = Image(value=image_bytes)
  assert img.model_dump() == b64encode(image_bytes).decode()

  # Test base64 string serialization
  b64_str = 'dGVzdCBiYXNlNjQgc3RyaW5n'
  img = Image(value=b64_str)
  assert img.model_dump() == b64decode(b64_str).decode()
