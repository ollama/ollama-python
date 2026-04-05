"""Basic chat completion.

Sends a single user message and prints the model's response.

Prerequisites:
    ollama pull gemma3

Usage:
    python chat.py
"""

from ollama import chat

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

response = chat('gemma3', messages=messages)
print(response['message']['content'])
