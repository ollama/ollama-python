from ollama import chat


def heading(text):
  print(text)
  print('=' * len(text))


messages = [
  {'role': 'user', 'content': 'What is 10 + 23?'},
]

# gpt-oss supports 'low', 'medium', 'high'
levels = ['low', 'medium', 'high']
for i, level in enumerate(levels):
  response = chat('gpt-oss:20b', messages=messages, think=level)

  heading(f'Thinking ({level})')
  print(response.message.thinking)
  print('\n')
  heading('Response')
  print(response.message.content)
  print('\n')
  if i < len(levels) - 1:
    print('-' * 20)
    print('\n')
