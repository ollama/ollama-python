from ollama import generate

response = generate('gemma4', 'Why is the sky blue?')
print(response['response'])
