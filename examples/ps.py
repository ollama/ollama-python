import ollama

# Ensure at least one model is loaded
progress_states = set()
for progress in ollama.pull('llama3.2', stream=True):
  if progress.get('status') in progress_states:
    continue
  progress_states.add(progress.get('status'))
  print(progress.get('status'))

print('\n')

print('Waiting for model to load... \n')
ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])

for model in ollama.ps().models:
  print('Model: ', model.model)
  print('  Digest: ', model.digest)
  print('  Expires at: ', model.expires_at)
  print('  Size: ', model.size)
  print('  Size vram: ', model.size_vram)
  print('  Details: ', model.details)
  print('\n')
