import shutil
import asyncio
import argparse

import ollama


async def speak(speaker, content):
  if speaker:
    p = await asyncio.create_subprocess_exec(speaker, content)
    await p.communicate()


async def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--speak', default=False, action='store_true')
  args = parser.parse_args()

  speaker = None
  if not args.speak:
    ...
  elif say := shutil.which('say'):
    speaker = say
  elif (espeak := shutil.which('espeak')) or (espeak := shutil.which('espeak-ng')):
    speaker = espeak

  client = ollama.AsyncClient()

  messages = []

  while True:
    if content_in := input('>>> '):
      messages.append({'role': 'user', 'content': content_in})

      content_out = ''
      message = {'role': 'assistant', 'content': ''}
      async for response in await client.chat(model='mistral', messages=messages, stream=True):
        if response['done']:
          messages.append(message)

        content = response['message']['content']
        print(content, end='', flush=True)

        content_out += content
        if content in ['.', '!', '?', '\n']:
          await speak(speaker, content_out)
          content_out = ''

        message['content'] += content

      if content_out:
        await speak(speaker, content_out)
      print()


try:
  asyncio.run(main())
except (KeyboardInterrupt, EOFError):
  ...
