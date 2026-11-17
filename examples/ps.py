from ollama import ProcessResponse, ps

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
