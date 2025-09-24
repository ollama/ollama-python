from ollama import generate

response = generate('gemma3', 'Why is the sky blue?')
print(response['response'])
