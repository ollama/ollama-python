from ollama import chat


options = {"temperature": 0.1, "seed": 100}

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

response = chat('mistral', messages=messages, options=options)
print(response['message']['content'])
