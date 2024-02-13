from ollama import historic_chat

message = {
  'role': 'user',
  'content': 'Tell me a Joke in less than 30 words.',
}
response = historic_chat('mistral', message=message)
print(response['message']['content'])

message = {
  'role': 'user',
  'content': 'Another please.',
}
response = historic_chat('mistral', message=message)
print(response['message']['content'])
