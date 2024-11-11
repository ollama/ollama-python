import sys

import pytest

if sys.version_info < (3, 10):
  pytest.skip('Python 3.10 or higher is required', allow_module_level=True)


from ollama._utils import _get_json_type, convert_function_to_tool, process_tools


def test_json_type_conversion():
  from typing import Optional

  # Test basic types
  assert _get_json_type(str) == 'string'
  assert _get_json_type(int) == 'integer'
  assert _get_json_type(list) == 'array'
  assert _get_json_type(dict) == 'object'

  # Test Optional
  assert _get_json_type(Optional[str]) == 'string'


def test_function_to_tool_conversion():
  from typing import Optional

  def add_numbers(x: int, y: Optional[int] = None) -> int:
    """Add two numbers together.
    Args:
        x (integer): The first number
        y (integer, optional): The second number

    Returns:
        integer: The sum of x and y
    """

    return x + y

  tool = convert_function_to_tool(add_numbers)

  assert tool['type'] == 'function'
  assert tool['function']['name'] == 'add_numbers'
  assert tool['function']['description'] == 'Add two numbers together.'
  assert tool['function']['parameters']['type'] == 'object'
  assert tool['function']['parameters']['properties']['x']['type'] == 'integer'
  assert tool['function']['parameters']['properties']['x']['description'] == 'The first number'
  assert tool['function']['parameters']['required'] == ['x']


def test_function_with_no_args():
  def simple_func():
    """
    A simple function with no arguments.
    Args:
        None
    Returns:
        None
    """
    pass

  tool = convert_function_to_tool(simple_func)
  assert tool.function.name == 'simple_func'
  assert tool.function.description == 'A simple function with no arguments.'
  assert tool.function.parameters.properties == {}
  assert tool.function.return_type is None


def test_function_with_all_types():
  def all_types(
    x: int,
    y: str,
    z: list[int],
    w: dict[str, int],
    v: int | str | None,
  ) -> int | dict[str, int] | str | list[int] | None:
    """
    A function with all types.
    Args:
        x (integer): The first number
        y (string): The second number
        z (array): The third number
        w (object): The fourth number
        v (integer | string | None): The fifth number
    """
    pass

  tool = convert_function_to_tool(all_types)
  assert tool.function.parameters.properties['x']['type'] == 'integer'
  assert tool.function.parameters.properties['y']['type'] == 'string'
  assert tool.function.parameters.properties['z']['type'] == 'array'
  assert tool.function.parameters.properties['w']['type'] == 'object'
  assert set(tool.function.parameters.properties['v']['type']) == {'string', 'integer'}
  assert set(tool.function.return_type) == {'string', 'integer', 'array', 'object'}


def test_process_tools():
  def func1(x: int) -> str:
    """Simple function 1.
    Args:
        x: A number
    """
    pass

  def func2(y: str) -> int:
    """Simple function 2.
    Args:
        y: A string
    """
    pass

  # Test with list of functions
  tools = process_tools([func1, func2])
  assert len(tools) == 2
  assert tools[0].function.name == 'func1'
  assert tools[1].function.name == 'func2'

  # Test with empty input
  assert process_tools() == []
  assert process_tools(None) == []
  assert process_tools([]) == []

  # Test with mix of functions and tool dicts
  tool_dict = {'type': 'function', 'function': {'name': 'test', 'description': 'Test function', 'parameters': {'type': 'object', 'properties': {'x': {'type': 'string', 'description': 'A string'}}, 'required': ['x']}}}
  tools = process_tools([func1, tool_dict])
  assert len(tools) == 2
  assert tools[0].function.name == 'func1'
  assert tools[1].function.name == 'test'
