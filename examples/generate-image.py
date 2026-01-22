# Image generation is experimental and currently only available on macOS

import base64

from ollama import generate

prompt = 'a sunset over mountains'
print(f'Prompt: {prompt}')

response = generate(
  model='x/z-image-turbo',
  prompt=prompt,
  width=1024,
  height=768,
)

# Save the generated image
with open('output.png', 'wb') as f:
  f.write(base64.b64decode(response.image))

print('Image saved to output.png')
