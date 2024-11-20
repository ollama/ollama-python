from ollama import embed

response = embed(model='llama3.1', input='Hello, world!')
print(response['embeddings'])
