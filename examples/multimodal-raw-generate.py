import sys
import random
import httpx

from ollama import generate

latest = httpx.get('https://xkcd.com/info.0.json')
latest.raise_for_status()

if len(sys.argv) > 1:
  num = int(sys.argv[1])
else:
  num = random.randint(1, latest.json().get('num'))

comic = httpx.get(f'https://xkcd.com/{num}/info.0.json')
comic.raise_for_status()

print(f'xkcd #{comic.json().get("num")}: {comic.json().get("alt")}')
print(f'link: https://xkcd.com/{num}')
print('---')

raw = httpx.get(comic.json().get('img'))
raw.raise_for_status()

for response in generate(model='llama3.2-vision', 
                         prompt='explain this comic:', 
                         raw=True, 
                         prepend_images_to_raw_prompt=True, 
                         images=[raw.content], 
                         stream=True):
  print(response['response'], end='', flush=True)

## Or to manually place image in specific location in prompt, use the below. 
## note: the `[img-0]<image>` tag can be placed anywhere in the prompt, and the `0` corresponds to the index of the image in the `images` array

# for response in generate(model='llama3.2-vision', 
#                          prompt='[img-0]<image>explain this comic:', 
#                          raw=True, 
#                          images=[raw.content], 
#                          stream=True):
#   print(response['response'], end='', flush=True)
