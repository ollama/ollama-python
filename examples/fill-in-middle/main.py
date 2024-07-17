from ollama import generate

response = generate(
  model='deepseek-coder-v2',
  prompt='def add(',
  suffix='return c',
)

print(response['response'])
