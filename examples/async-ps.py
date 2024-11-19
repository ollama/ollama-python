import asyncio
from ollama import AsyncClient


async def main():
  client = AsyncClient()

  response = await client.pull('llama3.1', stream=True)
  progress_states = set()
  async for progress in response:
    if progress.get('status') in progress_states:
      continue
    progress_states.add(progress.get('status'))
    print(progress.get('status'))

  print('\n')

  response = await client.chat('llama3.1', messages=[{'role': 'user', 'content': 'Hello!'}])
  print(response['message']['content'])

  print('\n')

  response = await client.ps()

  name = response['models'][0]['name']
  size = response['models'][0]['size']
  size_vram = response['models'][0]['size_vram']

  if size == size_vram:
    print(f'{name}: 100% GPU')
  elif not size_vram:
    print(f'{name}: 100% CPU')
  else:
    size_cpu = size - size_vram
    cpu_percent = round(size_cpu / size * 100)
    print(f'{name}: {cpu_percent}% CPU/{100 - cpu_percent}% GPU')


if __name__ == '__main__':
  asyncio.run(main())
