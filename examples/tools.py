from ollama import ChatResponse, chat


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
model = 'llama3.2'

response: ChatResponse = chat(
  model=model,
  messages=messages,
  tools=[add_two_numbers, subtract_two_numbers],
)

if response.message.tool_calls:
  # There may be multiple tool calls in the response
  for tool in response.message.tool_calls:
    # Ensure the function is available, and then call it
    if function_to_call := available_functions.get(tool.function.name):
      print('Calling function:', tool.function.name)
      print('Arguments:', tool.function.arguments)
      output = function_to_call(**tool.function.arguments)
      print('Function output:', output)
      messages.append(response.message.model_dump())
      messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})
    else:
      print('Function', tool.function.name, 'not found')

  # Get final response from model with function outputs
  final_response = chat(model=model, messages=messages)
  print('Final response:', final_response.message.content)

else:
  print('Response:', response.message.content)
