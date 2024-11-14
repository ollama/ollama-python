from base64 import b64decode, b64encode
import sys
from typing import Dict, List

import pytest

from ollama._types import Image, get_json_type


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
  assert get_json_type(List) == 'array'
  assert get_json_type(Dict) == 'object'


def test_advanced_json_type_conversion():
  from typing import Optional, Union, List, Dict, Sequence, Mapping, Set, Tuple, Any

  # Test nested collections
  assert get_json_type(List[List[int]]) == 'array'
  assert get_json_type(Dict[str, List[int]]) == 'object'

  # Test multiple unions
  result = get_json_type(Union[int, str, float])
  # Remove brackets from start/end
  result = result[1:-1] if result.startswith('[') else result
  assert set(x.strip().strip("'") for x in result.split(',')) == {'integer', 'string', 'number'}

  # Test collections.abc types
  assert get_json_type(Sequence[int]) == 'array'
  assert get_json_type(Mapping[str, int]) == 'object'
  assert get_json_type(Set[int]) == 'array'
  assert get_json_type(Tuple[int, str]) == 'array'

  # Test nested optionals
  assert get_json_type(Optional[List[Optional[int]]]) == 'array'

  # Test edge cases
  assert get_json_type(Any) == 'string'  # or however you want to handle Any
  assert get_json_type(None) == 'null'
  assert get_json_type(type(None)) == 'null'

  # Test complex nested types
  complex_type = Dict[str, Union[List[int], Optional[str], Dict[str, bool]]]
  assert get_json_type(complex_type) == 'object'


def test_invalid_types():
  # Test that invalid types raise appropriate errors
  with pytest.raises(ValueError):
    get_json_type(lambda x: x)  # Function type

  with pytest.raises(ValueError):
    get_json_type(type)  # metaclass


if sys.version_info >= (3, 10):

  def test_json_type_conversion_with_optional():
    from typing import Optional

    # Test basic types
    assert get_json_type(str) == 'string'
    assert get_json_type(int) == 'integer'
    assert get_json_type(list) == 'array'
    assert get_json_type(dict) == 'object'

    # Test Optional
    assert get_json_type(Optional[str]) == 'string'
