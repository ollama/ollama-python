from ollama import chat


messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for message in chat('mistral', messages=messages, stream=True):
  if message := message.get('message'):
    if message.get('role') == 'assistant':
      print(message.get('content', ''), end='', flush=True)

# end with a newline
print()
