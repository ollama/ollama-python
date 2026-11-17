import asyncio

import ollama


async def main():
  async with ollama.AsyncClient() as client:
    response = await client.generate('gemma4', 'Why is the sky blue?')
    print(response['response'])


if __name__ == '__main__':
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print('\nGoodbye!')
