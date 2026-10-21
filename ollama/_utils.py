from __future__ import annotations

import inspect
import re
from collections import defaultdict
from functools import partial
from typing import Callable, Union

import pydantic

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


def convert_function_to_tool(func: Callable) -> Tool:
  name_source = doc_source = func
  while isinstance(name_source, partial) and not hasattr(name_source, '__name__'):
    name_source = name_source.func
  while isinstance(doc_source, partial) and '__doc__' not in vars(doc_source):
    doc_source = doc_source.func

  # A partial's __wrapped__ metadata must not hide its bound arguments from inspect.
  signature_source = func
  bound_partials = []
  while isinstance(signature_source, partial) and getattr(signature_source, '__signature__', None) is None:
    bound_partials.append(signature_source)
    signature_source = signature_source.func
  for bound in reversed(bound_partials):
    signature_source = partial(signature_source, *bound.args, **bound.keywords)
  signature = inspect.signature(signature_source)

  doc_string = inspect.getdoc(doc_source)
  doc_string_hash = str(hash(doc_string))
  parsed_docstring = _parse_docstring(doc_string)
  schema = type(
    name_source.__name__,
    (pydantic.BaseModel,),
    {
      '__annotations__': {k: v.annotation if v.annotation != inspect._empty else str for k, v in signature.parameters.items()},
      '__signature__': signature,
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
      name=name_source.__name__,
      description=schema.get('description', ''),
      parameters=Tool.Function.Parameters(**schema),
    ),
  )

  return Tool.model_validate(tool)
