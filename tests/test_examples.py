import runpy
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import ollama


@pytest.mark.parametrize('example', ['multi-tool.py', 'gpt-oss-tools-stream.py'])
def test_streamed_tools_preserve_one_complete_assistant_turn(monkeypatch, example):
  requests = []
  monkeypatch.setitem(sys.modules, 'rich', SimpleNamespace(print=print))

  class Client:
    def chat(self, *args, messages, **kwargs):
      requests.append(deepcopy(messages))
      if len(requests) > 1:
        return iter([ollama.ChatResponse(message={'role': 'assistant', 'content': 'Done'})])
      return iter(
        [
          ollama.ChatResponse(message={'role': 'assistant', 'thinking': 'First ', 'tool_calls': [{'function': {'name': 'unknown_one', 'arguments': {}}}]}),
          ollama.ChatResponse(message={'role': 'assistant', 'thinking': 'second', 'content': 'Working ', 'tool_calls': [{'function': {'name': 'unknown_two', 'arguments': {}}}]}),
          ollama.ChatResponse(message={'role': 'assistant', 'content': 'now'}),
        ]
      )

  monkeypatch.setattr(ollama, 'Client', Client)
  runpy.run_path(str(Path(__file__).parents[1] / 'examples' / example))
  assert len(requests) == 2
  history = requests[1]
  assert [message['role'] for message in history] == ['user', 'assistant', 'tool', 'tool']
  assert history[1]['thinking'] == 'First second'
  assert history[1]['content'] == 'Working now'
  assert [call.function.name for call in history[1]['tool_calls']] == ['unknown_one', 'unknown_two']
  assert [message['tool_name'] for message in history[2:]] == ['unknown_one', 'unknown_two']


@pytest.mark.parametrize('example,method', [('chat-logprobs.py', 'chat'), ('generate-logprobs.py', 'generate')])
def test_logprob_examples_accept_missing_alternatives(monkeypatch, capsys, example, method):
  response_type = ollama.ChatResponse if method == 'chat' else ollama.GenerateResponse
  fields = {'message': {'role': 'assistant', 'content': 'Hi'}} if method == 'chat' else {'response': 'Hi'}
  response = response_type(**fields, logprobs=[{'token': 'Hi', 'logprob': -0.1}])
  monkeypatch.setattr(ollama, method, lambda **kwargs: response)
  runpy.run_path(str(Path(__file__).parents[1] / 'examples' / example))
  assert 'logprob=-0.100' in capsys.readouterr().out
  # A server may omit logprobs altogether too.
  response.logprobs = None
  runpy.run_path(str(Path(__file__).parents[1] / 'examples' / example))
