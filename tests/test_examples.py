from pathlib import Path
from runpy import run_path

import pytest

import ollama


@pytest.mark.parametrize(
  ('logprobs', 'expected'),
  [
    (None, 'chat logprobs:'),
    ([{'token': 'hi', 'logprob': -0.1}], 'logprob=-0.100'),
    (
      [{'token': 'hi', 'logprob': -0.1, 'top_logprobs': [{'token': 'hey', 'logprob': -0.2}]}],
      "alt -> 'hey'",
    ),
  ],
)
def test_chat_logprobs_example(logprobs, expected, monkeypatch, capsys):
  response = ollama.ChatResponse.model_validate({'message': {'role': 'assistant', 'content': 'hi'}, 'logprobs': logprobs})
  monkeypatch.setattr(ollama, 'chat', lambda **_: response)

  run_path(str(Path(__file__).resolve().parents[1] / 'examples' / 'chat-logprobs.py'))

  output = capsys.readouterr().out
  assert 'Chat response: hi' in output
  assert expected in output
