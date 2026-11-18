import pytest

from ollama import Message, Tool


@pytest.mark.parametrize('key', ['content', 'model_dump', 'nonexistent'])
def test_get_missing_key_uses_default(key):
  message = Message(role='assistant')
  fallback = object()

  with pytest.raises(KeyError):
    message[key]
  assert message.get(key, fallback) is fallback


@pytest.mark.parametrize('content', [None, ''])
def test_get_preserves_explicit_values(content):
  message = Message(role='assistant', content=content)
  assert message.get('content', 'fallback') == content


def test_get_preserves_model_default():
  assert Tool().get('type', 'fallback') == 'function'
