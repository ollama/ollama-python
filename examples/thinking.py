from ollama import chat

messages = [
  {
    'role': 'user',
    'content': 'What is 10 + 23?',
  },
]

response = chat('deepseek-r1', messages=messages, think=True)

print('Thinking:\n========\n\n' + response.message.thinking)
print('\nResponse:\n========\n\n' + response.message.content)
