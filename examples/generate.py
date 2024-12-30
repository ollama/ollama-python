import ollama

response = ollama.generate('llama3.2', 'Why is the sky blue?')
print(response['response'])
