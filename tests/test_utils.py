import json
import sys

import pytest

if sys.version_info < (3, 10):
  pytest.skip('Python 3.10 or higher is required', allow_module_level=True)


from ollama._utils import convert_function_to_tool


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

  tool = convert_function_to_tool(add_numbers).model_dump()

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

  tool = convert_function_to_tool(simple_func).model_dump()
  assert tool['function']['name'] == 'simple_func'
  assert tool['function']['description'] == 'A simple function with no arguments.'
  assert tool['function']['parameters']['properties'] == {}


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

  tool_json = convert_function_to_tool(all_types).model_dump_json()
  tool = json.loads(tool_json)
  assert tool['function']['parameters']['properties']['x']['type'] == 'integer'
  assert tool['function']['parameters']['properties']['y']['type'] == 'string'
  assert tool['function']['parameters']['properties']['z']['type'] == 'array'
  assert tool['function']['parameters']['properties']['w']['type'] == 'object'
  assert set(x.strip().strip("'") for x in tool['function']['parameters']['properties']['v']['type'].removeprefix('[').removesuffix(']').split(',')) == {'string', 'integer'}


def test_function_docstring_parsing():
  from typing import List, Dict, Any

  def func_with_complex_docs(x: int, y: List[str]) -> Dict[str, Any]:
    """
    Test function with complex docstring.

    Args:
        x (integer): A number
            with multiple lines
        y (array of string): A list
            with multiple lines

    Returns:
        object: A dictionary
            with multiple lines
    """
    pass

  tool = convert_function_to_tool(func_with_complex_docs).model_dump()
  assert tool['function']['description'] == 'Test function with complex docstring.'
  assert tool['function']['parameters']['properties']['x']['description'] == 'A number with multiple lines'
  assert tool['function']['parameters']['properties']['y']['description'] == 'A list with multiple lines'


def test_skewed_docstring_parsing():
  def add_two_numbers(x: int, y: int) -> int:
    """
    Add two numbers together.
    Args:
        x (integer):: The first number




        y (integer ): The second number
    Returns:
        integer: The sum of x and y
    """
    pass

  tool = convert_function_to_tool(add_two_numbers).model_dump()
  assert tool['function']['parameters']['properties']['x']['description'] == ': The first number'
  assert tool['function']['parameters']['properties']['y']['description'] == 'The second number'
