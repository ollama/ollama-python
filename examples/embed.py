"""Generate text embeddings.

Produces a vector embedding for the given input text, useful for
semantic search, clustering, and similarity comparisons.

Prerequisites:
    ollama pull llama3.2

Usage:
    python embed.py
"""

from ollama import embed

response = embed(model='llama3.2', input='Hello, world!')
print(response['embeddings'])
