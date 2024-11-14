from base64 import b64decode, b64encode

import pytest


from ollama._types import Image, Tool


def test_image_serialization():
  # Test bytes serialization
  image_bytes = b'test image bytes'
  img = Image(value=image_bytes)
  assert img.model_dump() == b64encode(image_bytes).decode()

  # Test base64 string serialization
  b64_str = 'dGVzdCBiYXNlNjQgc3RyaW5n'
  img = Image(value=b64_str)
  assert img.model_dump() == b64decode(b64_str).decode()


def test_tool_serialization():
  # Test valid tool serialization
  tool = Tool(
    function=Tool.Function(
      name='add_two_numbers',
      description='Add two numbers together.',
      parameters=Tool.Function.Parameters(
        type='object',
        properties={
          'a': Tool.Function.Parameters.Property(
            type=int,
            description='The first number',
          ),
          'b': Tool.Function.Parameters.Property(
            type=int,
            description='The second number',
          ),
        },
        required=['a', 'b'],
      ),
    )
  )
  assert tool.model_dump() == {
    'type': 'function',
    'function': {
      'name': 'add_two_numbers',
      'description': 'Add two numbers together.',
      'parameters': {
        'type': 'object',
        'properties': {
          'a': {
            'type': 'integer',
            'description': 'The first number',
          },
          'b': {
            'type': 'integer',
            'description': 'The second number',
          },
        },
        'required': ['a', 'b'],
      },
    },
  }

  # Test invalid type
  with pytest.raises(ValueError):
    property = Tool.Function.Parameters.Property(
      type=lambda x: x,  # Invalid type
      description='Invalid type',
    )
    Tool.model_validate(
      Tool(
        function=Tool.Function(
          parameters=Tool.Function.Parameters(
            properties={
              'x': property,
            }
          )
        )
      )
    )

  # Test invalid parameters type
  with pytest.raises(ValueError):
    Tool.model_validate(
      Tool(
        function=Tool.Function(
          name='test',
          description='Test',
          parameters=Tool.Function.Parameters(
            type='invalid_type',  # Must be 'object'
            properties={},
          ),
        )
      )
    )
