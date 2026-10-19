from __future__ import annotations

import inspect
import re
from collections import defaultdict
from typing import Callable, Container, Union

import pydantic

from ollama._types import Tool


def _parse_docstring(doc_string: Union[str, None], parameters: Container[str] = ()) -> dict[str, str]:
  parsed_docstring = defaultdict(str)
  if not doc_string:
    return parsed_docstring

  key = str(hash(doc_string))
  args_indent = 0
  for line in doc_string.splitlines():
    lowered_line = line.lower().strip()
    indent = len(line) - len(line.lstrip())
    name, _, text = line.strip().partition(':')
    # A deeper line under Args like `raises: ...` documents the parameter of that exact name; bare or other headers still end Args
    is_parameter = key == 'args' and indent > args_indent and name in parameters and text.strip() != ''
    if lowered_line.startswith('args:'):
      key = 'args'
      args_indent = indent
    elif lowered_line.startswith(('returns:', 'yields:', 'raises:')) and not is_parameter:
      key = '_'

    else:
      # maybe change to a list and join later
      parsed_docstring[key] += f'{line.strip()}\n'

  last_key = None
  for line in parsed_docstring['args'].splitlines():
    line = line.strip()
    if ':' in line:
      # Split the line on either:
      # 1. A parenthetical expression like (integer) - captured in group 1
      # 2. A colon :
      # Followed by optional whitespace. Only split on first occurrence.
      parts = re.split(r'(?:\(([^)]*)\)|:)\s*', line, maxsplit=1)

      arg_name = parts[0].strip()
      last_key = arg_name

      # Get the description - will be in parts[1] if parenthetical or parts[-1] if after colon
      arg_description = parts[-1].strip()
      if len(parts) > 2 and parts[1]:  # Has parenthetical content
        arg_description = parts[-1].split(':', 1)[-1].strip()

      parsed_docstring[last_key] = arg_description

    elif last_key and line:
      parsed_docstring[last_key] += ' ' + line

  return parsed_docstring


def convert_function_to_tool(func: Callable) -> Tool:
  doc_string_hash = str(hash(inspect.getdoc(func)))
  parsed_docstring = _parse_docstring(inspect.getdoc(func), inspect.signature(func).parameters)
  schema = type(
    func.__name__,
    (pydantic.BaseModel,),
    {
      '__annotations__': {k: v.annotation if v.annotation != inspect._empty else str for k, v in inspect.signature(func).parameters.items()},
      '__signature__': inspect.signature(func),
      '__doc__': parsed_docstring[doc_string_hash],
    },
  ).model_json_schema()

  for k, v in schema.get('properties', {}).items():
    # If type is missing, the default is string
    types = {t.get('type', 'string') for t in v.get('anyOf')} if 'anyOf' in v else {v.get('type', 'string')}
    if 'null' in types:
      schema['required'].remove(k)
      types.discard('null')

    schema['properties'][k] = {
      'description': parsed_docstring[k],
      'type': ', '.join(types),
    }

  tool = Tool(
    type='function',
    function=Tool.Function(
      name=func.__name__,
      description=schema.get('description', ''),
      parameters=Tool.Function.Parameters(**schema),
    ),
  )

  return Tool.model_validate(tool)
