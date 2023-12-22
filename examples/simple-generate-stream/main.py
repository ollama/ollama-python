from ollama import generate


for part in generate('mistral', 'Why is the sky blue?', stream=True):
  print(part['response'], end='', flush=True)
