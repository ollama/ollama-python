"""Basic text generation.

Generates a response from a prompt without conversation history.

Prerequisites:
    ollama pull gemma3

Usage:
    python generate.py
"""

from ollama import generate

response = generate('gemma3', 'Why is the sky blue?')
print(response['response'])
