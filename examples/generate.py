from ollama import generate

response = generate('llama3.2', 'Why is the sky blue?')
print(response['response'])
