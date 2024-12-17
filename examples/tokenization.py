import ollama

# Get tokens from a model
response = ollama.tokenize(model='llama3.2', text='Why the sky is blue?')
tokens = response.tokens
print('Tokens from model', tokens)

# Convert tokens back to text
response = ollama.detokenize(model='llama3.2', tokens=tokens)
print('Text from tokens', response.text)  # Prints: Why the sky is blue?
