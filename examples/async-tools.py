import asyncio

import ollama
from ollama import ChatResponse


def add_two_numbers(a: int, b: int) -> int:
  """
  Add two numbers

  Args:
    a (int): The first number
    b (int): The second number

  Returns:
    int: The sum of the two numbers
  """
  return a + b


def subtract_two_numbers(a: int, b: int) -> int:
  """
  Subtract two numbers
  """
  return a - b


# Tools can still be manually defined and passed into chat
subtract_two_numbers_tool = {
  'type': 'function',
  'function': {
    'name': 'subtract_two_numbers',
    'description': 'Subtract two numbers',
    'parameters': {
      'type': 'object',
      'required': ['a', 'b'],
      'properties': {
        'a': {'type': 'integer', 'description': 'The first number'},
        'b': {'type': 'integer', 'description': 'The second number'},
      },
    },
  },
}

messages = [{'role': 'user', 'content': 'What is three plus one?'}]
print('Prompt:', messages[0]['content'])

available_functions = {
  'add_two_numbers': add_two_numbers,
  'subtract_two_numbers': subtract_two_numbers,
}


async def main():
  client = ollama.AsyncClient()

  response: ChatResponse = await client.chat(
    'llama3.1',
    messages=messages,
    tools=[add_two_numbers, subtract_two_numbers_tool],
  )

  if response.message.tool_calls:
    # There may be multiple tool calls in the response
    # Add the assistant message with tool calls to the conversation
    messages.append(response.message)

    for tool in response.message.tool_calls:
      # Ensure the function is available, and then call it
      if function_to_call := available_functions.get(tool.function.name):
        print('Calling function:', tool.function.name)
        print('Arguments:', tool.function.arguments)
        output = function_to_call(**tool.function.arguments)
        print('Function output:', output)
      else:
        print('Function', tool.function.name, 'not found')
        output = 'Function not found'

      # Add each tool result as a separate message
      messages.append({'role': 'tool', 'content': str(output), 'tool_name': tool.function.name})

    # Get final response from model with all tool call results
    final_response = await client.chat('llama3.1', messages=messages)
    print('Final response:', final_response.message.content)

  else:
    print('No tool calls returned from model')


if __name__ == '__main__':
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print('\nGoodbye!')
