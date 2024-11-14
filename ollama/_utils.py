from __future__ import annotations

from icecream import ic
from typing import Any, Callable, Mapping, Optional, Union, get_args, get_origin
from ollama._types import Tool
from collections.abc import Sequence
from typing import Dict
import sys

# Type compatibility layer for Union and UnionType
if sys.version_info >= (3, 10):
  from types import UnionType

  def is_union(tp: Any) -> bool:
    return get_origin(tp) in (Union, UnionType)
else:

  def is_union(tp: Any) -> bool:
    return get_origin(tp) is Union


# # Map both the type and the type reference to the same JSON type
# TYPE_MAP = {
#   # Basic types
#   int: 'integer',
#   'int': 'integer',
#   str: 'string',
#   'str': 'string',
#   float: 'number',
#   'float': 'number',
#   bool: 'boolean',
#   'bool': 'boolean',
#   type(None): 'null',
#   None: 'null',
#   'None': 'null',
#   # Collection types
#   list: 'array',
#   'list': 'array',
#   List: 'array',
#   'List': 'array',
#   Sequence: 'array',
#   'Sequence': 'array',
#   tuple: 'array',
#   'tuple': 'array',
#   set: 'array',
#   'set': 'array',
#   Set: 'array',
#   TypeSet: 'array',
#   'Set': 'array',
#   # Mapping types
#   dict: 'object',
#   'dict': 'object',
#   Dict: 'object',
#   'Dict': 'object',
#   Mapping: 'object',
#   'Mapping': 'object',
#   Any: 'string',
#   'Any': 'string',
# }


# def _get_json_type(python_type: Any) -> str | List[str]:
#   # Handle Optional types (Union[type, None] and type | None)
#   if is_union(python_type):
#     args = get_args(python_type)
#     # Filter out None/NoneType from union args
#     if non_none_args := [arg for arg in args if arg not in (None, type(None))]:
#       if len(non_none_args) == 1:
#         return _get_json_type(non_none_args[0])
#       # For multiple return types (e.g., int | str | None), return stringified array of types -> "['integer', 'string', 'null']"
#       return str([_get_json_type(arg) for arg in non_none_args]).replace(' ', '')
#     return 'null'

#   # Handle generic types (List[int], Dict[str, int], etc.)
#   if get_origin(python_type) is not None:
#     # Get the base type (List, Dict, etc.)
#     base_type = TYPE_MAP.get(get_origin(python_type), None)
#     if base_type:
#       return base_type
#     # If it's a subclass of known abstract base classes, map to appropriate type
#     if isinstance(get_origin(python_type), type):
#       if issubclass(get_origin(python_type), (list, Sequence, tuple, set, Set)):
#         return 'array'
#       if issubclass(get_origin(python_type), (dict, Mapping)):
#         return 'object'

#   # Handle both type objects and type references (older Python versions)
#   type_key = python_type
#   if isinstance(python_type, type):
#     type_key = python_type
#   elif isinstance(python_type, str):
#     type_key = python_type

#   # If type not found in map, try to get the type name
#   if type_key not in TYPE_MAP and hasattr(python_type, '__name__'):
#     type_key = python_type.__name__

#   if type_key in TYPE_MAP:
#     return TYPE_MAP[type_key]

#   raise ValueError(f'Could not map Python type {python_type} to a valid JSON type')


def _is_optional_type(python_type: Any) -> bool:
  if is_union(python_type):
    args = get_args(python_type)
    return any(arg in (None, type(None)) for arg in args)
  return False


def _parse_docstring(func: Callable, doc_string: str) -> tuple[str, Dict[str, str]]:
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

  # Parse parameter descriptions
  param_descriptions = {}
  current_param = None
  param_desc_lines = []
  indent_level = None

  for line in args_section.split('\n'):
    stripped_line = line.strip()
    if not stripped_line:
      continue

    # Check for new parameter
    for param_name in func.__annotations__:
      if param_name == 'return':
        continue
      if stripped_line.startswith(f'{param_name}:') or stripped_line.startswith(f'{param_name} ') or stripped_line.startswith(f'{param_name}('):
        # Save previous parameter if exists
        if current_param:
          param_descriptions[current_param] = ' '.join(param_desc_lines).strip()
          param_desc_lines = []

        current_param = param_name
        # Get description after parameter name
        desc_part = stripped_line.split(':', 1)[1].strip() if ':' in stripped_line else ''
        if desc_part:
          param_desc_lines.append(desc_part)
        indent_level = len(line) - len(line.lstrip())
        break
    else:
      # Handle continuation lines
      if current_param and line.startswith(' ' * (indent_level + 4 if indent_level else 0)):
        param_desc_lines.append(stripped_line)
      elif current_param and stripped_line:
        # Different indentation means new parameter
        param_descriptions[current_param] = ' '.join(param_desc_lines).strip()
        param_desc_lines = []
        current_param = None

  # Save last parameter
  if current_param:
    param_descriptions[current_param] = ' '.join(param_desc_lines).strip()

  # Verify all parameters have descriptions
  for param_name in func.__annotations__:
    if param_name == 'return':
      continue
    if param_name not in param_descriptions:
      raise ValueError(f'Parameter {param_name} must have a description in the Args section')

  return description, param_descriptions


def convert_function_to_tool(func: Callable) -> Tool:
  doc_string = func.__doc__
  if not doc_string:
    raise ValueError(f'Function {func.__name__} must have a docstring in Google format. Example:\n' '"""Add two numbers.\n\n' 'Args:\n' '    a: First number\n' '    b: Second number\n\n' 'Returns:\n' '    int: Sum of the numbers\n' '"""')

  description, param_descriptions = _parse_docstring(func, doc_string)

  parameters = Tool.Function.Parameters(type='object', properties={}, required=[])

  for param_name, param_type in func.__annotations__.items():
    if param_name == 'return':
      continue

    # parameters.properties[param_name] = Tool.Function.Parameters.Property(type=_get_json_type(param_type), description=param_descriptions[param_name])
    ic(param_type)
    ic(type(param_type))
    ic(is_union(param_type))
    parameters.properties[param_name] = Tool.Function.Parameters.Property(type=param_type, description=param_descriptions[param_name])

    # Only add to required if not optional
    if not _is_optional_type(param_type):
      parameters.required.append(param_name)

  function = Tool.Function(name=func.__name__, description=description, parameters=parameters, return_type=None)

  # Check if consumable by ollama/ollama
  # if 'return' in func.__annotations__ and func.__annotations__['return'] is not None:
  #   function.return_type = _get_json_type(func.__annotations__['return'])

  tool = Tool(function=function)

  ic(tool.type)
  ic(tool.function.name)
  ic(tool.function.description)
  ic(tool.function.parameters.type)
  ic(tool.function.parameters.required)
  ic(tool.function.parameters.properties)
  ic(tool.function.return_type)
  return tool


def _process_tools(tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None) -> Sequence[Tool]:
  if not tools:
    return []

  processed_tools = []
  for tool in tools:
    if callable(tool):
      processed_tools.append(convert_function_to_tool(tool))
    else:
      processed_tools.append(Tool.model_validate(tool))

  return processed_tools
