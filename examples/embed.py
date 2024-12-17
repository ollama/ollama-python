import ollama

response = ollama.embed(model='llama3.2', input='Hello, world!')
print(response['embeddings'])
