from ollama import generate


response = generate('mistral', 'Why is the sky blue?')
print(response['response'])
