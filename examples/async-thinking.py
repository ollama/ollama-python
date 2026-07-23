import asyncio

from ollama import AsyncClient


async def main():
  messages = [
    {
      'role': 'user',
      'content': 'How many r letters are in the word strawberry?',
    },
  ]

  client = AsyncClient()
  response = await client.chat('deepseek-r1', messages=messages, think=True)

  print('Thinking:\n========\n\n' + response.message.thinking)
  print('\nResponse:\n========\n\n' + response.message.content)


if __name__ == '__main__':
  asyncio.run(main())
