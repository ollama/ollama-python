from ollama import ChatResponse, chat, create_function_tool
from ollama import get_ollama_tool_description

def add_two_numbers(a: int, b: int) -> int:
  """
  Add two numbers

  Args:
    a (int): The first number
    b (int): The second number

  Returns:
    int: The sum of the two numbers
  """

  # The cast is necessary as returned tool call arguments don't always conform exactly to schema
  # E.g. this would prevent "what is 30 + 12" to produce '3012' instead of 42
  return int(a) + int(b)


def subtract_two_numbers(a: int, b: int) -> int:
  """
  Subtract two numbers
  """

  # The cast is necessary as returned tool call arguments don't always conform exactly to schema
  return int(a) - int(b)

def multiply_two_numbers(a: int, b: int) -> int:
  """
  Multiply two numbers
  """
  return int(a) * int(b)

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

# A simple way to define tools manually, even though it seems long
multiply_two_numbers_tool = create_function_tool(tool_name="multiply_two_numbers", 
                                                 description="Multiply two numbers", 
                                                 parameter_list=[{"a": {"type": "integer", "description": "The first number"}, 
                                                                  "b": {"type": "integer", "description": "The second number"}}], 
                                                 required_parameters=["a", "b"])

messages = [
  {'role': 'system', 'content': f'You are a helpful assistant, with access to these tools: {get_ollama_tool_description()}'}, #usage example for the get_ollama_tool_description function
  {'role': 'user', 'content': 'What is three plus one? and Search the web for what is ollama'}]
print('Prompt:', messages[1]['content'])

available_functions = {
  'add_two_numbers': add_two_numbers,
  'subtract_two_numbers': subtract_two_numbers,
  'multiply_two_numbers': multiply_two_numbers,
}

response: ChatResponse = chat(
  'llama3.1',
  messages=messages,
  tools=[add_two_numbers, subtract_two_numbers_tool, multiply_two_numbers_tool],
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
    else:
      print('Function', tool.function.name, 'not found')

# Only needed to chat with the model using the tool call results
if response.message.tool_calls:
  # Add the function response to messages for the model to use
  messages.append(response.message)
  messages.append({'role': 'tool', 'content': str(output), 'tool_name': tool.function.name})

  # Get final response from model with function outputs
  final_response = chat('llama3.1', messages=messages)
  print('Final response:', final_response.message.content)

else:
  print('No tool calls returned from model')
