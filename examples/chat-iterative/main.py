from ollama import iterative_chat

message = {
  'role': 'user',
  'content': 'Tell me a Joke in less than 30 words.',
}
response = iterative_chat('mistral', message=message)
print(response['message']['content'])

message = {
  'role': 'user',
  'content': 'Another please.',
}
response = iterative_chat('mistral', message=message)
print(response['message']['content'])
