from ollama import embed

response = embed(model='embeddinggemma', input='Hello, world!')
print(response['embeddings'])
