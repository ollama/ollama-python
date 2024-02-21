from ollama import iterative_chat


message = {
  'role': 'user',
  'content': 'Tell me a Joke in less than 30 words.',
}

for part in iterative_chat('llama2:70b', message=message, stream=True):
    print(part['message']['content'], end='', flush=True)

message = {
  'role': 'user',
  'content': 'Another please.',
}

for part in iterative_chat('llama2:70b', message=message, stream=True):
    print(part['message']['content'], end='', flush=True)