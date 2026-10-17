from __future__ import annotations

import inspect
import re
from collections import defaultdict
from typing import Callable, Union

import pydantic
from typing_extensions import get_type_hints

from ollama._types import Tool


def _parse_docstring(doc_string: Union[str, None]) -> dict[str, str]:
  parsed_docstring = defaultdict(str)
  if not doc_string:
    return parsed_docstring

  key = str(hash(doc_string))
  for line in doc_string.splitlines():
    lowered_line = line.lower().strip()
    if lowered_line.startswith('args:'):
      key = 'args'
    elif lowered_line.startswith(('returns:', 'yields:', 'raises:')):
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


def _resolve_type_hints(func: Callable) -> dict:
  # String annotations (quoted, or under `from __future__ import annotations`) name types
  # from the function's module, which pydantic cannot see from here, so evaluate them there.
  try:
    return get_type_hints(inspect.unwrap(func), include_extras=True)
  except Exception:
    return {}


def _parameter_annotation(name: str, parameter: inspect.Parameter, type_hints: dict):
  if parameter.annotation is inspect.Parameter.empty:
    return str
  if isinstance(parameter.annotation, str):
    return type_hints.get(name, parameter.annotation)
  return parameter.annotation


def convert_function_to_tool(func: Callable) -> Tool:
  doc_string_hash = str(hash(inspect.getdoc(func)))
  parsed_docstring = _parse_docstring(inspect.getdoc(func))
  type_hints = _resolve_type_hints(func)
  schema = type(
    func.__name__,
    (pydantic.BaseModel,),
    {
      '__annotations__': {k: _parameter_annotation(k, v, type_hints) for k, v in inspect.signature(func).parameters.items()},
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
