from ollama import historic_chat
from ollama import _client


message = {
  'role': 'user',
  'content': 'Tell me a Joke in less than 30 words.',
}

for part in historic_chat('mistral', message=message, stream=True):
    print(part['message']['content'], end='', flush=True)

message = {
  'role': 'user',
  'content': 'Another please.',
}

for part in historic_chat('mistral', message=message, stream=True):
    print(part['message']['content'], end='', flush=True)