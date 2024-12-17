import ollama

for part in ollama.generate('llama3.2', 'Why is the sky blue?', stream=True):
  print(part['response'], end='', flush=True)
