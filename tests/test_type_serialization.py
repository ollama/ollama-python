from base64 import b64encode


from ollama._types import Image


def test_image_serialization(tmp_path):
  # Test bytes serialization
  image_bytes = b'test image bytes'
  img = Image(value=image_bytes)
  assert img.model_dump() == b64encode(image_bytes).decode()

  # Test base64 string serialization
  b64_str = 'dGVzdCBiYXNlNjQgc3RyaW5n'
  img = Image(value=b64_str)
  assert img.model_dump() == b64_str  # Should return as-is if valid base64

  # Test regular string that's not a path
  img = Image(value='not a path or base64')
  assert img.model_dump() == 'not a path or base64'  # Should return as-is

  # # Test regular string path serialization
  # test_file2 = tmp_path / 'test2.txt'
  # test_file2.write_bytes(b'test file content')
  # img = Image(value=str(test_file2))
  # assert img.model_dump() == b64encode(test_file2.read_bytes()).decode()

  # # Test Path serialization
  # test_file = tmp_path / 'test.txt'
  # test_file.write_bytes(b'test file content')
  # img = Image(value=test_file)
  # assert img.model_dump() == b64encode(test_file.read_bytes()).decode()
