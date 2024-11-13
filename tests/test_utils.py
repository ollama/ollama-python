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
  assert set(x.strip().strip("'") for x in tool.function.parameters.properties['v']['type'].removeprefix('[').removesuffix(']').split(',')) == {'string', 'integer'}
  assert set(x.strip().strip("'") for x in tool.function.return_type.removeprefix('[').removesuffix(']').split(',')) == {'string', 'integer', 'array', 'object'}


def test_process_tools():
  def func1(x: int) -> str:
    """Simple function 1.
    Args:
        x (integer): A number
    """
    pass

  def func2(y: str) -> int:
    """Simple function 2.
    Args:
        y (string): A string
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


def test_advanced_json_type_conversion():
  from typing import Optional, Union, List, Dict, Sequence, Mapping, Set, Tuple, Any

  # Test nested collections
  assert _get_json_type(List[List[int]]) == 'array'
  assert _get_json_type(Dict[str, List[int]]) == 'object'

  # Test multiple unions
  assert set(x.strip().strip("'") for x in _get_json_type(Union[int, str, float]).removeprefix('[').removesuffix(']').split(',')) == {'integer', 'string', 'number'}

  # Test collections.abc types
  assert _get_json_type(Sequence[int]) == 'array'
  assert _get_json_type(Mapping[str, int]) == 'object'
  assert _get_json_type(Set[int]) == 'array'
  assert _get_json_type(Tuple[int, str]) == 'array'

  # Test nested optionals
  assert _get_json_type(Optional[List[Optional[int]]]) == 'array'

  # Test edge cases
  assert _get_json_type(Any) == 'string'  # or however you want to handle Any
  assert _get_json_type(None) == 'null'
  assert _get_json_type(type(None)) == 'null'

  # Test complex nested types
  complex_type = Dict[str, Union[List[int], Optional[str], Dict[str, bool]]]
  assert _get_json_type(complex_type) == 'object'


def test_invalid_types():
  # Test that invalid types raise appropriate errors
  with pytest.raises(ValueError):
    _get_json_type(lambda x: x)  # Function type

  with pytest.raises(ValueError):
    _get_json_type(type)  # metaclass


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

  tool = convert_function_to_tool(func_with_complex_docs)
  assert tool['function']['description'] == 'Test function with complex docstring.'
  assert tool['function']['parameters']['properties']['x']['description'] == 'A number with multiple lines'
  assert tool['function']['parameters']['properties']['y']['description'] == 'A list with multiple lines'


def test_tool_validation():
  # Test that malformed tool dictionaries are rejected
  invalid_tool = {'type': 'invalid_type', 'function': {'name': 'test'}}
  with pytest.raises(ValueError):
    process_tools([invalid_tool])

  # Test missing required fields
  incomplete_tool = {
    'type': 'function',
    'function': {'name': 'test'},  # missing description and parameters
  }
  with pytest.raises(ValueError):
    process_tools([incomplete_tool])
