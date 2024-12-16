import ollama

# Get tokens from a model
response = ollama.tokenize(model='llama3.2', text='Hello world!')
tokens = response.tokens
print('tokens from model', tokens)

# Convert tokens back to text
response = ollama.detokenize(model='llama3.2', tokens=tokens)
print('text from tokens', response.text)  # Prints: Hello world!
