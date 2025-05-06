import json
import sys
from typing import Dict, List, Mapping, Sequence, Set, Tuple, Union

from ollama._utils import convert_function_to_tool


def test_function_to_tool_conversion():
  def add_numbers(x: int, y: Union[int, None] = None) -> int:
    """Add two numbers together.
    args:
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

  tool = convert_function_to_tool(simple_func).model_dump()
  assert tool['function']['name'] == 'simple_func'
  assert tool['function']['description'] == 'A simple function with no arguments.'
  assert tool['function']['parameters']['properties'] == {}


def test_function_with_all_types():
  if sys.version_info >= (3, 10):

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
  else:

    def all_types(
      x: int,
      y: str,
      z: Sequence,
      w: Mapping[str, int],
      d: Dict[str, int],
      s: Set[int],
      t: Tuple[int, str],
      l: List[int],  # noqa: E741
      o: Union[int, None],
    ) -> Union[Mapping[str, int], str, None]:
      """
      A function with all types.
      Args:
          x (integer): The first number
          y (string): The second number
          z (array): The third number
          w (object): The fourth number
          d (object): The fifth number
          s (array): The sixth number
          t (array): The seventh number
          l (array): The eighth number
          o (integer | None): The ninth number
      """

  tool_json = convert_function_to_tool(all_types).model_dump_json()
  tool = json.loads(tool_json)
  assert tool['function']['parameters']['properties']['x']['type'] == 'integer'
  assert tool['function']['parameters']['properties']['y']['type'] == 'string'

  if sys.version_info >= (3, 10):
    assert tool['function']['parameters']['properties']['z']['type'] == 'array'
    assert tool['function']['parameters']['properties']['w']['type'] == 'object'
    assert {x.strip().strip("'") for x in tool['function']['parameters']['properties']['v']['type'].removeprefix('[').removesuffix(']').split(',')} == {'string', 'integer'}
    assert tool['function']['parameters']['properties']['v']['type'] != 'null'
    assert tool['function']['parameters']['required'] == ['x', 'y', 'z', 'w']
  else:
    assert tool['function']['parameters']['properties']['z']['type'] == 'array'
    assert tool['function']['parameters']['properties']['w']['type'] == 'object'
    assert tool['function']['parameters']['properties']['d']['type'] == 'object'
    assert tool['function']['parameters']['properties']['s']['type'] == 'array'
    assert tool['function']['parameters']['properties']['t']['type'] == 'array'
    assert tool['function']['parameters']['properties']['l']['type'] == 'array'
    assert tool['function']['parameters']['properties']['o']['type'] == 'integer'
    assert tool['function']['parameters']['properties']['o']['type'] != 'null'
    assert tool['function']['parameters']['required'] == ['x', 'y', 'z', 'w', 'd', 's', 't', 'l']


def test_function_docstring_parsing():
  from typing import Any, Dict, List

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

  tool = convert_function_to_tool(func_with_complex_docs).model_dump()
  assert tool['function']['description'] == 'Test function with complex docstring.'
  assert tool['function']['parameters']['properties']['x']['description'] == 'A number with multiple lines'
  assert tool['function']['parameters']['properties']['y']['description'] == 'A list with multiple lines'


def test_skewed_docstring_parsing():
  def add_two_numbers(x: int, y: int) -> int:
    """
    Add two numbers together.
    Args:
        x (integer): : The first number




        y (integer ): The second number
    Returns:
        integer: The sum of x and y
    """

  tool = convert_function_to_tool(add_two_numbers).model_dump()
  assert tool['function']['parameters']['properties']['x']['description'] == ': The first number'
  assert tool['function']['parameters']['properties']['y']['description'] == 'The second number'


def test_function_with_no_docstring():
  def no_docstring(): ...

  def no_docstring_with_args(x: int, y: int): ...

  tool = convert_function_to_tool(no_docstring).model_dump()
  assert tool['function']['description'] == ''

  tool = convert_function_to_tool(no_docstring_with_args).model_dump()
  assert tool['function']['description'] == ''
  assert tool['function']['parameters']['properties']['x']['description'] == ''
  assert tool['function']['parameters']['properties']['y']['description'] == ''


def test_function_with_only_description():
  def only_description():
    """
    A function with only a description.
    """

  tool = convert_function_to_tool(only_description).model_dump()
  assert tool['function']['description'] == 'A function with only a description.'
  assert tool['function']['parameters'] == {'type': 'object', 'defs': None, 'items': None, 'required': None, 'properties': {}}

  def only_description_with_args(x: int, y: int):
    """
    A function with only a description.
    """

  tool = convert_function_to_tool(only_description_with_args).model_dump()
  assert tool['function']['description'] == 'A function with only a description.'
  assert tool['function']['parameters'] == {
    'type': 'object',
    'defs': None,
    'items': None,
    'properties': {
      'x': {'type': 'integer', 'description': '', 'enum': None, 'items': None},
      'y': {'type': 'integer', 'description': '', 'enum': None, 'items': None},
    },
    'required': ['x', 'y'],
  }


def test_function_with_yields():
  def function_with_yields(x: int, y: int):
    """
    A function with yields section.

    Args:
      x: the first number
      y: the second number

    Yields:
      The sum of x and y
    """

  tool = convert_function_to_tool(function_with_yields).model_dump()
  assert tool['function']['description'] == 'A function with yields section.'
  assert tool['function']['parameters']['properties']['x']['description'] == 'the first number'
  assert tool['function']['parameters']['properties']['y']['description'] == 'the second number'


def test_function_with_no_types():
  def no_types(a, b):
    """
    A function with no types.
    """

  tool = convert_function_to_tool(no_types).model_dump()
  assert tool['function']['parameters']['properties']['a']['type'] == 'string'
  assert tool['function']['parameters']['properties']['b']['type'] == 'string'


def test_function_with_parentheses():
  def func_with_parentheses(a: int, b: int) -> int:
    """
    A function with parentheses.
    Args:
        a: First (:thing) number to add
        b: Second number to add
    Returns:
        int: The sum of a and b
    """

  def func_with_parentheses_and_args(a: int, b: int):
    """
    A function with parentheses and args.
    Args:
        a(integer) : First (:thing) number to add
        b(integer) :Second number to add
    """

  tool = convert_function_to_tool(func_with_parentheses).model_dump()
  assert tool['function']['parameters']['properties']['a']['description'] == 'First (:thing) number to add'
  assert tool['function']['parameters']['properties']['b']['description'] == 'Second number to add'

  tool = convert_function_to_tool(func_with_parentheses_and_args).model_dump()
  assert tool['function']['parameters']['properties']['a']['description'] == 'First (:thing) number to add'
  assert tool['function']['parameters']['properties']['b']['description'] == 'Second number to add'
