from ollama import chat

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in chat('gemma3', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)
