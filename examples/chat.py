import ollama

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

response = ollama.chat('llama3.2', messages=messages)
print(response['message']['content'])
