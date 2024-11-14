from __future__ import annotations
from typing import Callable
from ollama._types import Tool, _is_optional_type
from typing import Dict


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

    parameters.properties[param_name] = Tool.Function.Parameters.Property(type=param_type, description=param_descriptions[param_name])

    # Only add to required if not optional
    if not _is_optional_type(param_type):
      parameters.required.append(param_name)

  function = Tool.Function(name=func.__name__, description=description, parameters=parameters, return_type=None)

  tool = Tool(function=function)

  return tool
