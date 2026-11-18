import base64

import pytest

from ollama._types import Image


def test_image_data_uri_serializes_as_base64():
  encoded = base64.b64encode(b'example image bytes').decode()
  assert Image(value=f'data:image/png;base64,{encoded}').model_dump() == encoded
  assert Image(value=f'DATA:IMAGE/PNG;BASE64,{encoded}').model_dump() == encoded


@pytest.mark.parametrize(
  'value',
  [
    'data:image/png,not-base64',
    'data:image/png;base64,invalid!',
    'data:image/png;base64,',
    'data:text/plain;base64,ZXhhbXBsZQ==',
  ],
)
def test_invalid_image_data_uri_raises(value: str):
  with pytest.raises(ValueError, match='Invalid image data URI'):
    Image(value=value).model_dump()
