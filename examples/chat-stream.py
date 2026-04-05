"""Streaming chat response.

Prints tokens as they are generated instead of waiting for the full
response, providing a real-time typing effect.

Prerequisites:
    ollama pull gemma3

Usage:
    python chat-stream.py
"""

from ollama import chat

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in chat('gemma3', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)
