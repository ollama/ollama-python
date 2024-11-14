from base64 import b64decode, b64encode
import sys
from typing import Dict, List

import pytest

from ollama._types import Image, _get_json_type


def test_image_serialization():
  # Test bytes serialization
  image_bytes = b'test image bytes'
  img = Image(value=image_bytes)
  assert img.model_dump() == b64encode(image_bytes).decode()

  # Test base64 string serialization
  b64_str = 'dGVzdCBiYXNlNjQgc3RyaW5n'
  img = Image(value=b64_str)
  assert img.model_dump() == b64decode(b64_str).decode()


def test_json_type_conversion():
  # Test basic types
  assert _get_json_type(List) == 'array'
  assert _get_json_type(Dict) == 'object'


def test_advanced_json_type_conversion():
  from typing import Optional, Union, List, Dict, Sequence, Mapping, Set, Tuple, Any

  # Test nested collections
  assert _get_json_type(List[List[int]]) == 'array'
  assert _get_json_type(Dict[str, List[int]]) == 'object'

  # Test multiple unions
  assert set(x.strip().strip("'") for x in _get_json_type(Union[int, str, float]).removeprefix('[').removesuffix(']').split(',')) == {'integer', 'string', 'number'}

  # Test collections.abc types
  assert _get_json_type(Sequence[int]) == 'array'
  assert _get_json_type(Mapping[str, int]) == 'object'
  assert _get_json_type(Set[int]) == 'array'
  assert _get_json_type(Tuple[int, str]) == 'array'

  # Test nested optionals
  assert _get_json_type(Optional[List[Optional[int]]]) == 'array'

  # Test edge cases
  assert _get_json_type(Any) == 'string'  # or however you want to handle Any
  assert _get_json_type(None) == 'null'
  assert _get_json_type(type(None)) == 'null'

  # Test complex nested types
  complex_type = Dict[str, Union[List[int], Optional[str], Dict[str, bool]]]
  assert _get_json_type(complex_type) == 'object'


def test_invalid_types():
  # Test that invalid types raise appropriate errors
  with pytest.raises(ValueError):
    _get_json_type(lambda x: x)  # Function type

  with pytest.raises(ValueError):
    _get_json_type(type)  # metaclass


if sys.version_info >= (3, 10):

  def test_json_type_conversion_with_optional():
    from typing import Optional

    # Test basic types
    assert _get_json_type(str) == 'string'
    assert _get_json_type(int) == 'integer'
    assert _get_json_type(list) == 'array'
    assert _get_json_type(dict) == 'object'

    # Test Optional
    assert _get_json_type(Optional[str]) == 'string'
