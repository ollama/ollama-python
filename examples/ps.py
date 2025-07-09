from ollama import ProcessResponse, chat, ps, pull

# Ensure at least one model is loaded
response = pull('llama3.2', stream=True)
progress_states = set()
for progress in response:
  if progress.get('status') in progress_states:
    continue
  progress_states.add(progress.get('status'))
  print(progress.get('status'))

print('\n')

print('Waiting for model to load... \n')
chat(model='llama3.2', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])


response: ProcessResponse = ps()
for model in response.models:
  print('Model: ', model.model)
  print('  Digest: ', model.digest)
  print('  Expires at: ', model.expires_at)
  print('  Size: ', model.size)
  print('  Size vram: ', model.size_vram)
  print('  Details: ', model.details)
  print('  Context length: ', model.context_length)
  print('\n')
