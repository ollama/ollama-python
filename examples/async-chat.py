import asyncio
import ollama


async def main():
  client = ollama.AsyncClient()
  messages = []

  print("Chat with the model (type 'exit' to quit)")
  while True:
    # Get user input
    user_input = input('You: ')
    if user_input.lower() == 'exit':
      break

    # Add user message to history
    messages.append({'role': 'user', 'content': user_input})

    # Stream the response
    print('Assistant: ', end='', flush=True)
    async for chunk in await client.chat(model='llama3.1', messages=messages, stream=True):
      # Print the response chunk
      print(chunk['message']['content'], end='', flush=True)

      # Update message history with complete response
      if chunk['done']:
        messages.append(chunk['message'])
    print('\n')


if __name__ == '__main__':
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print('\nGoodbye!')
