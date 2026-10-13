import runpy
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ollama import ChatResponse


@pytest.fixture
def anyio_backend():
  return 'asyncio'


@pytest.mark.anyio
@pytest.mark.parametrize(
  ('function_name', 'arguments', 'expected'),
  [
    ('add_two_numbers', {'a': '30', 'b': '12'}, '42'),
    ('subtract_two_numbers', {'a': '30', 'b': '12'}, '18'),
    ('add_two_numbers', {'a': 30, 'b': 12}, '42'),
    ('subtract_two_numbers', {'a': 30, 'b': 12}, '18'),
  ],
)
async def test_async_tools_example_returns_numeric_tool_results(function_name, arguments, expected):
  example = runpy.run_path(str(Path(__file__).parents[1] / 'examples' / 'async-tools.py'))
  tool_response = ChatResponse(
    message={
      'role': 'assistant',
      'tool_calls': [{'function': {'name': function_name, 'arguments': arguments}}],
    },
  )
  chat = AsyncMock(side_effect=[tool_response, ChatResponse(message={'role': 'assistant', 'content': 'Done'})])

  with patch('ollama.AsyncClient') as client:
    client.return_value.chat = chat
    await example['main']()

  assert chat.await_count == 2
  assert chat.await_args_list[1].kwargs['messages'] == [
    {'role': 'user', 'content': 'What is three plus one?'},
    tool_response.message,
    {'role': 'tool', 'content': expected, 'tool_name': function_name},
  ]
