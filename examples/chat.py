from ollama import chat

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

response = chat('gemma3', messages=messages)
print(response['message']['content'])
