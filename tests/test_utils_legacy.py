from typing import Dict, List, Mapping, Sequence, Set, Tuple, Union

from ollama._utils import _get_json_type, convert_function_to_tool


def test_json_type_conversion():
  # Test basic types
  assert _get_json_type(List) == 'array'
  assert _get_json_type(Dict) == 'object'


def test_function_to_tool_conversion():
  def add_numbers(x: int, y: Union[int, None] = None) -> int:
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


def test_function_with_all_typing_types():
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
    pass

  tool = convert_function_to_tool(all_types)
  assert tool.function.parameters.properties['x']['type'] == 'integer'
  assert tool.function.parameters.properties['y']['type'] == 'string'
  assert tool.function.parameters.properties['z']['type'] == 'array'
  assert tool.function.parameters.properties['w']['type'] == 'object'
  assert tool.function.parameters.properties['d']['type'] == 'object'
  assert tool.function.parameters.properties['s']['type'] == 'array'
  assert tool.function.parameters.properties['t']['type'] == 'array'
  assert tool.function.parameters.properties['l']['type'] == 'array'
  assert tool.function.parameters.properties['o']['type'] == 'integer'
  assert set(x.strip().strip("'") for x in tool.function.return_type[1:-1].split(',')) == {'string', 'object'}
