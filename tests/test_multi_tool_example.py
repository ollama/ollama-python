import copy
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from ollama import ChatResponse


@pytest.mark.parametrize('count', [1, 2])
def test_multi_tool_example_records_one_assistant_for_multiple_tools(monkeypatch, capsys, count):
  requests = []
  tool_names = ['get_temperature', 'get_conditions'][:count]
  tool_calls = [{'function': {'name': name, 'arguments': {'city': 'London'}}} for name in tool_names]
  responses = [
    [ChatResponse(message={'role': 'assistant', 'tool_calls': tool_calls})],
    [ChatResponse(message={'role': 'assistant', 'content': 'The weather is mild.'})],
  ]

  def chat(*args, **kwargs):
    requests.append(copy.deepcopy(kwargs['messages']))
    return iter(responses.pop(0))

  monkeypatch.setattr('ollama.Client', lambda: SimpleNamespace(chat=chat))

  runpy.run_path(str(Path(__file__).parents[1] / 'examples' / 'multi-tool.py'))

  assert len(requests) == 2
  assert [message['role'] for message in requests[1]] == ['user', 'assistant', *(['tool'] * count)]
  assert [message['tool_name'] for message in requests[1][2:]] == tool_names
  assert len(requests[1][1]['tool_calls']) == count
  assert 'The weather is mild.' in capsys.readouterr().out


def test_multi_tool_example_without_tools_does_not_request_another_response(monkeypatch, capsys):
  requests = []

  def chat(*args, **kwargs):
    requests.append(kwargs)
    return iter([ChatResponse(message={'role': 'assistant', 'content': 'Hello.'})])

  monkeypatch.setattr('ollama.Client', lambda: SimpleNamespace(chat=chat))

  runpy.run_path(str(Path(__file__).parents[1] / 'examples' / 'multi-tool.py'))

  assert len(requests) == 1
  assert 'No tool calls returned' in capsys.readouterr().out
