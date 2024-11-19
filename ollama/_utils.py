from __future__ import annotations
from collections import defaultdict
import inspect
from typing import Callable, Union

import pydantic
from ollama._types import Tool


def _parse_docstring(doc_string: Union[str, None]) -> dict[str, str]:
  parsed_docstring = defaultdict(str)
  if not doc_string:
    return parsed_docstring

  lowered_doc_string = doc_string.lower()

  key = hash(doc_string)
  parsed_docstring[key] = ''
  for line in lowered_doc_string.splitlines():
    if line.startswith('args:'):
      key = 'args'
    elif line.startswith('returns:') or line.startswith('yields:') or line.startswith('raises:'):
      key = '_'

    else:
      # maybe change to a list and join later
      parsed_docstring[key] += f'{line.strip()}\n'

  last_key = None
  for line in parsed_docstring['args'].splitlines():
    line = line.strip()
    if ':' in line and not line.startswith('args'):
      # Split on first occurrence of '(' or ':' to separate arg name from description
      split_char = '(' if '(' in line else ':'
      arg_name, rest = line.split(split_char, 1)

      last_key = arg_name.strip()
      # Get description after the colon
      arg_description = rest.split(':', 1)[1].strip() if split_char == '(' else rest.strip()
      parsed_docstring[last_key] = arg_description

    elif last_key and line:
      parsed_docstring[last_key] += ' ' + line

  return parsed_docstring


def convert_function_to_tool(func: Callable) -> Tool:
  doc_string_hash = hash(inspect.getdoc(func))
  parsed_docstring = _parse_docstring(inspect.getdoc(func))
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
    function=Tool.Function(
      name=func.__name__,
      description=schema.get('description', ''),
      parameters=Tool.Function.Parameters(**schema),
    )
  )

  return Tool.model_validate(tool)
