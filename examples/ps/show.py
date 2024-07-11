from ollama import show, pull

import json

def prettify_json(data):
    pretty_json = json.dumps(data, indent=4)
    return pretty_json

response = pull('llava', stream=True)
progress_states = set()
for progress in response:
  if progress.get('status') in progress_states:
    continue
  progress_states.add(progress.get('status'))
  print(progress.get('status'))

print('\n')

response = show('llava')
print(prettify_json(response))
