import sys
from typing import Any, List, Mapping, Optional, Sequence, Union, get_origin, get_args
from collections.abc import Set
from typing import Dict, Set as TypeSet, TypeVar

T = TypeVar('T')
if sys.version_info >= (3, 10):
  from types import UnionType
else:
  UnionType = Union[T]

# Python doesn't have a type serializer, so we need to map types to JSON types
TYPE_MAP = {
  # Basic types
  int: 'integer',
  'int': 'integer',
  'integer': 'integer',
  str: 'string',
  'str': 'string',
  'string': 'string',
  float: 'number',
  'float': 'number',
  'number': 'number',
  bool: 'boolean',
  'bool': 'boolean',
  'boolean': 'boolean',
  type(None): 'null',
  None: 'null',
  'None': 'null',
  'null': 'null',
  # Collection types
  list: 'array',
  'list': 'array',
  List: 'array',
  'List': 'array',
  Sequence: 'array',
  'Sequence': 'array',
  tuple: 'array',
  'tuple': 'array',
  set: 'array',
  'set': 'array',
  Set: 'array',
  TypeSet: 'array',
  'Set': 'array',
  'array': 'array',
  # Mapping types
  dict: 'object',
  'dict': 'object',
  Dict: 'object',
  'Dict': 'object',
  Mapping: 'object',
  'Mapping': 'object',
  'object': 'object',
  Any: 'string',
  'Any': 'string',
}

if sys.version_info >= (3, 10):
  from types import UnionType

  def is_union(tp: Any) -> bool:
    return get_origin(tp) in (Union, UnionType)
else:

  def is_union(tp: Any) -> bool:
    return get_origin(tp) is Union


def _map_type(python_type: Any) -> str:
  # Handle generic types (List[int], Dict[str, int], etc.)
  origin = get_origin(python_type)
  if origin is not None:
    # Get the base type (List, Dict, etc.)
    base_type = TYPE_MAP.get(origin, None)
    if base_type:
      return base_type
    # If it's a subclass of known abstract base classes, map to appropriate type
    if isinstance(origin, type):
      if issubclass(origin, (list, Sequence, tuple, set, Set)):
        return 'array'
      if issubclass(origin, (dict, Mapping)):
        return 'object'

  # Handle both type objects and type references (older Python versions)
  type_key = python_type
  if isinstance(python_type, type):
    type_key = python_type
  elif isinstance(python_type, str):
    type_key = python_type

  # If type not found in map, try to get the type name
  if type_key not in TYPE_MAP and hasattr(python_type, '__name__'):
    type_key = python_type.__name__

  if type_key in TYPE_MAP:
    return TYPE_MAP[type_key]

  raise ValueError(f'Could not map Python type {python_type} to a valid JSON type')


def get_json_type(python_type: Union[type, UnionType, Optional[T]]) -> Union[str, List[str]]:
  # Handle Optional types (Union[type, None] and type | None)
  if is_union(python_type):
    args = get_args(python_type)
    # Filter out None/NoneType from union args
    if non_none_args := [arg for arg in args if arg not in (None, type(None))]:
      if len(non_none_args) == 1:
        return _map_type(non_none_args[0])
      # For multiple return types (e.g., int | str | None), return stringified array of types -> "['integer', 'string', 'null']"
      return str([_map_type(arg) for arg in non_none_args]).replace(' ', '')
  return _map_type(python_type)
