from ollama import generate

response = generate('deepseek-r1', 'why is the sky blue', think=True)

print('Thinking:\n========\n\n' + response.thinking)
print('\nResponse:\n========\n\n' + response.response)
