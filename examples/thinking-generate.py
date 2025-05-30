from ollama import generate

messages = [
  {
    'role': 'user',
    'content': 'What is 10 + 23?',
  },
]

response = generate('deepseek-r1', 'why is the sky blue', think=True)

print('Thinking:\n========\n\n' + response.thinking)
print('\nResponse:\n========\n\n' + response.response)
