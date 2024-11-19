from ollama import ps, pull
from ollama._types import ProcessResponse

# Ensure at least one model is loaded
response = pull('llama3.1', stream=True)
progress_states = set()
for progress in response:
  if progress.get('status') in progress_states:
    continue
  progress_states.add(progress.get('status'))
  print(progress.get('status'))

print('\n')


response: ProcessResponse = ps()
for model in response.models:
  print(f'Model: {model.model}')
  print(f'Digest: {model.digest}')
  print(f'Expires at: {model.expires_at}')
  print(f'Size: {model.size}')
  print(f'Size vram: {model.size_vram}')
  print(f'Details: {model.details}')

  print('---' * 10)
