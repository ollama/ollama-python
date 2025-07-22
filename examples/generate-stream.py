from ollama import generate

for part in generate('gemma3', 'Why is the sky blue?', stream=True):
  print(part['response'], end='', flush=True)
