import asyncio

import ollama


async def main():
  messages = [
    {
      'role': 'user',
      'content': 'Why is the sky blue?',
    },
  ]

  client = ollama.AsyncClient()
  response = await client.chat('llama3.2', messages=messages)
  print(response['message']['content'])


if __name__ == '__main__':
  asyncio.run(main())
