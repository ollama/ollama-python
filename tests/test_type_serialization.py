from base64 import b64encode


from ollama._types import Image


def test_image_serialization():
  # Test bytes serialization
  image_bytes = b'test image bytes'
  img = Image(value=image_bytes)
  assert img.model_dump() == b64encode(image_bytes).decode()

  # Test Path serialization
  from pathlib import Path
  import tempfile

  with tempfile.NamedTemporaryFile() as f:
    f.write(b'test file content')
    f.flush()
    path = Path(f.name)
    img = Image(value=path)
    assert img.model_dump() == b64encode(path.read_bytes()).decode()

  # Test base64 string serialization
  b64_str = 'dGVzdCBiYXNlNjQgc3RyaW5n'
  img = Image(value=b64_str)
  assert img.model_dump() == b64_str  # Should return as-is if valid base64

  # Test regular string path serialization
  with tempfile.NamedTemporaryFile() as f:
    f.write(b'test file content')
    f.flush()
    img = Image(value=f.name)
    assert img.model_dump() == b64encode(Path(f.name).read_bytes()).decode()

  # Test regular string that's not a path
  img = Image(value='not a path or base64')
  assert img.model_dump() == 'not a path or base64'  # Should return as-is
