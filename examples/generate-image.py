# Image generation is experimental and currently only available on macOS

import base64

from ollama import generate

prompt = 'a sunset over mountains'
print(f'Prompt: {prompt}')

for response in generate(model='x/z-image-turbo', prompt=prompt):
  if response.image:
    # Final response contains the image
    with open('output.png', 'wb') as f:
      f.write(base64.b64decode(response.image))
    print('\nImage saved to output.png')
  elif response.total:
    # Progress update
    print(f'Progress: {response.completed}/{response.total}', end='\r')
