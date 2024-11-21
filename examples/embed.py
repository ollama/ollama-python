from ollama import embed

response = embed(model='llama3.2', input='Hello, world!')
print(response['embeddings'])
