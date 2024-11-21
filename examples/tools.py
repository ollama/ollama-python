from ollama import chat
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

prompt = 'What is three plus one?'
print(f'Prompt: {prompt}')

response: ChatResponse = chat(
  'llama3.1',
  messages=[{'role': 'user', 'content': prompt}],
  tools=[add_two_numbers, subtract_two_numbers_tool],
)

available_functions = {
  'add_two_numbers': add_two_numbers,
  'subtract_two_numbers': subtract_two_numbers,
}

if response.message.tool_calls:
  # There may be multiple tool calls in the response
  for tool in response.message.tool_calls:
    # Ensure the function is available, and then call it
    if tool.function.name in available_functions:
      print(f'Calling function {tool.function.name}')
      print(f'Arguments: {tool.function.arguments}')
      function_to_call = available_functions[tool.function.name]
      print(f'Function output: {function_to_call(**tool.function.arguments)}')
    else:
      print(f'Function {tool.function.name} not found')
