from __future__ import annotations
import inspect
from typing import Callable, Union

import pydantic
from ollama._types import Tool


def _parse_docstring(doc_string: Union[str, None]) -> dict[str, str]:
  parsed_docstring = {'description': ''}
  if not doc_string:
    return parsed_docstring

  lowered_doc_string = doc_string.lower()

  if 'args:' not in lowered_doc_string:
    parsed_docstring['description'] = lowered_doc_string.strip()
    return parsed_docstring

  else:
    parsed_docstring['description'] = lowered_doc_string.split('args:')[0].strip()
    args_section = lowered_doc_string.split('args:')[1]

  if 'returns:' in lowered_doc_string:
    # Return section can be captured and used
    args_section = args_section.split('returns:')[0]

  cur_var = None
  for line in args_section.split('\n'):
    line = line.strip()
    if not line:
      continue
    if ':' not in line:
      # Continuation of the previous parameter's description
      if cur_var:
        parsed_docstring[cur_var] += f' {line}'
      continue

    # For the case with: `param_name (type)`: ...
    if '(' in line:
      param_name = line.split('(')[0]
      param_desc = line.split('):')[1]

    # For the case with: `param_name: ...`
    else:
      param_name, param_desc = line.split(':', 1)

    parsed_docstring[param_name.strip()] = param_desc.strip()
    cur_var = param_name.strip()

  return parsed_docstring


def convert_function_to_tool(func: Callable) -> Tool:
  schema = type(
    func.__name__,
    (pydantic.BaseModel,),
    {
      '__annotations__': {k: v.annotation for k, v in inspect.signature(func).parameters.items()},
      '__signature__': inspect.signature(func),
      '__doc__': inspect.getdoc(func),
    },
  ).model_json_schema()

  properties = {}
  required = []
  parsed_docstring = _parse_docstring(schema.get('description'))
  for k, v in schema.get('properties', {}).items():
    properties[k] = {}
    properties[k]['description'] = parsed_docstring.get(k, '')
    if 'anyOf' in v:
      anyof_list = v['anyOf']
      properties[k]['type'] = [anyof_type.get('type', 'string') for anyof_type in anyof_list if anyof_type.get('type', 'string') != 'null']
      if len(properties[k]['type']) == 1:
        properties[k]['type'] = properties[k]['type'][0]
      else:
        properties[k]['type'] = str(properties[k]['type'])
    else:
      properties[k]['type'] = v.get('type', None)
      required.append(k)

  schema['properties'] = properties

  tool = Tool(
    function=Tool.Function(
      name=func.__name__,
      description=parsed_docstring.get('description'),
      parameters=Tool.Function.Parameters(
        type='object',
        properties=schema.get('properties', {}),
        required=required,
      ),
    )
  )
  print(tool.model_dump())

  return tool
