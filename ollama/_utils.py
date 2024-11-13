from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Union, get_args, get_origin
from ollama._types import Tool
from collections.abc import Sequence, Set
from typing import Dict, Set as TypeSet
import sys

# Type compatibility layer for Union and UnionType
if sys.version_info >= (3, 10):
  from types import UnionType

  def is_union(tp: Any) -> bool:
    return get_origin(tp) in (Union, UnionType)
else:

  def is_union(tp: Any) -> bool:
    return get_origin(tp) is Union


# Map both the type and the type reference to the same JSON type
TYPE_MAP = {
  # Basic types
  int: 'integer',
  'int': 'integer',
  str: 'string',
  'str': 'string',
  float: 'number',
  'float': 'number',
  bool: 'boolean',
  'bool': 'boolean',
  type(None): 'null',
  None: 'null',
  'None': 'null',
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
  # Mapping types
  dict: 'object',
  'dict': 'object',
  Dict: 'object',
  'Dict': 'object',
  Mapping: 'object',
  'Mapping': 'object',
  Any: 'string',
  'Any': 'string',
}


def _get_json_type(python_type: Any) -> str | List[str]:
  # Handle Optional types (Union[type, None] and type | None)
  if is_union(python_type):
    args = get_args(python_type)
    # Filter out None/NoneType from union args
    if non_none_args := [arg for arg in args if arg not in (None, type(None))]:
      if len(non_none_args) == 1:
        return _get_json_type(non_none_args[0])
      # For multiple return types (e.g., int | str | None), return stringified array of types -> "['integer', 'string', 'null']"
      return str([_get_json_type(arg) for arg in non_none_args]).replace(' ', '')
    return 'null'

  # Handle generic types (List[int], Dict[str, int], etc.)
  if get_origin(python_type) is not None:
    # Get the base type (List, Dict, etc.)
    base_type = TYPE_MAP.get(get_origin(python_type), None)
    if base_type:
      return base_type
    # If it's a subclass of known abstract base classes, map to appropriate type
    if isinstance(get_origin(python_type), type):
      if issubclass(get_origin(python_type), (list, Sequence, tuple, set, Set)):
        return 'array'
      if issubclass(get_origin(python_type), (dict, Mapping)):
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


def _is_optional_type(python_type: Any) -> bool:
  if is_union(python_type):
    args = get_args(python_type)
    return any(arg in (None, type(None)) for arg in args)
  return False


def convert_function_to_tool(func: Callable) -> Tool:
  doc_string = func.__doc__
  if not doc_string:
    raise ValueError(f'Function {func.__name__} must have a docstring in Google format. Example:\n' '"""Add two numbers.\n\n' 'Args:\n' '    a: First number\n' '    b: Second number\n\n' 'Returns:\n' '    int: Sum of the numbers\n' '"""')

  # Extract description from docstring - get all lines before Args:
  description_lines = []
  for line in doc_string.split('\n'):
    line = line.strip()
    if line.startswith('Args:'):
      break
    if line:
      description_lines.append(line)

  description = ' '.join(description_lines).strip()

  # Parse Args section
  if 'Args:' not in doc_string:
    raise ValueError(f'Function {func.__name__} docstring must have an Args section in Google format')

  args_section = doc_string.split('Args:')[1]
  if 'Returns:' in args_section:
    args_section = args_section.split('Returns:')[0]

  parameters = {'type': 'object', 'properties': {}, 'required': []}

  # Build parameters dict
  for param_name, param_type in func.__annotations__.items():
    if param_name == 'return':
      continue

    param_desc_lines = []
    found_param = False
    indent_level = None

    # Process docstring lines
    for line in args_section.split('\n'):
      stripped_line = line.strip()
      if not stripped_line:
        continue

      # Check for parameter start
      if stripped_line.startswith(f'{param_name}:') or stripped_line.startswith(f'{param_name} ') or stripped_line.startswith(f'{param_name}('):
        found_param = True
        # Get the description part after the parameter name
        desc_part = stripped_line.split(':', 1)[1].strip() if ':' in stripped_line else ''
        if desc_part:
          param_desc_lines.append(desc_part)
        # Get the indentation level for continuation lines
        indent_level = len(line) - len(line.lstrip())
        continue

      # Handle continuation lines
      if found_param and line.startswith(' ' * (indent_level + 4)):
        # Add continuation line, stripped of extra indentation
        param_desc_lines.append(stripped_line)
      elif found_param and stripped_line:
        # If we hit a line with different indentation, we're done with this parameter
        break

    if not found_param:
      raise ValueError(f'Parameter {param_name} must have a description in the Args section')

    # Join all lines with spaces
    param_desc = ' '.join(param_desc_lines).strip()

    parameters['properties'][param_name] = {
      'type': _get_json_type(param_type),
      'description': param_desc,
    }

    # Only add to required if not optional
    if not _is_optional_type(param_type):
      parameters['required'].append(param_name)

  tool_dict = {
    'type': 'function',
    'function': {
      'name': func.__name__,
      'description': description,
      'parameters': parameters,
      'return_type': None,
    },
  }

  if 'return' in func.__annotations__ and func.__annotations__['return'] is not None:
    tool_dict['function']['return_type'] = _get_json_type(func.__annotations__['return'])

  return Tool.model_validate(tool_dict)


def process_tools(tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None) -> Sequence[Tool]:
  if not tools:
    return []

  processed_tools = []
  for tool in tools:
    if callable(tool):
      processed_tools.append(convert_function_to_tool(tool))
    else:
      processed_tools.append(Tool.model_validate(tool))

  return processed_tools
