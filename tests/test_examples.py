import copy
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

from ollama import ChatResponse


def test_gpt_oss_stream_preserves_content_before_tool_result(monkeypatch):
  requests = []
  responses = [
    [
      ChatResponse(message={'role': 'assistant', 'content': 'Checking '}),
      ChatResponse(message={'role': 'assistant', 'content': 'London.', 'tool_calls': [{'function': {'name': 'get_weather', 'arguments': {'city': 'London'}}}]}),
    ],
    [ChatResponse(message={'role': 'assistant', 'content': 'The weather is mild.'})],
  ]

  def chat(**kwargs):
    requests.append(copy.deepcopy(kwargs['messages']))
    return iter(responses.pop(0))

  monkeypatch.setattr('ollama.Client', lambda: SimpleNamespace(chat=chat))
  monkeypatch.setitem(sys.modules, 'rich', SimpleNamespace(print=lambda *args, **kwargs: None))

  runpy.run_path(str(Path(__file__).parents[1] / 'examples' / 'gpt-oss-tools-stream.py'))

  assert len(requests) == 2
  assistant, tool = requests[1][1:]
  assert assistant['role'] == 'assistant'
  assert assistant['content'] == 'Checking London.'
  assert len(assistant['tool_calls']) == 1
  assert tool['role'] == 'tool'
  assert tool['tool_name'] == 'get_weather'
