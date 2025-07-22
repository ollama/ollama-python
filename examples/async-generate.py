import asyncio

import ollama


async def main():
  client = ollama.AsyncClient()
  response = await client.generate('gemma3', 'Why is the sky blue?')
  print(response['response'])


if __name__ == '__main__':
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print('\nGoodbye!')
