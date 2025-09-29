import asyncio
from ollama import ChatResponse, chat
from ollama import (
  ollama_tool, 
  ollama_async_tool, 
  get_ollama_tools, 
  get_ollama_name_async_tools, 
  get_ollama_tools_name, 
  get_ollama_tool_description)

@ollama_tool
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

@ollama_tool
def subtract_two_numbers(a: int, b: int) -> int:
  """
  Subtract two numbers
  Args:
    a (int): The first number
    b (int): The second number
  Returns:
    int: The difference of the two numbers
  """
  return a - b

@ollama_async_tool
async def web_search(query: str) -> str:
  """
  Search the web for information,
  Args:
    query (str): The query to search the web for
  Returns:
    str: The result of the web search
  """
  return f"Searching the web for {query}"

available_functions = get_ollama_tools_name() # this is a dictionary of tools

# tools are treated differently in synchronous code
async_available_functions = get_ollama_name_async_tools()

messages = [
  {'role': 'system', 'content': f'You are a helpful assistant, with access to these tools: {get_ollama_tool_description()}'}, #usage example for the get_ollama_tool_description function
  {'role': 'user', 'content': 'What is three plus one? and Search the web for what is ollama'}]
print('Prompt:', messages[1]['content'])

response: ChatResponse = chat(
  'llama3.1',
  messages=messages,
  tools=get_ollama_tools(), # this is the list of tools using decorators
)

if response.message.tool_calls:
  # There may be multiple tool calls in the response
  for tool in response.message.tool_calls:
    # Ensure the function is available, and then call it
    if function_to_call := available_functions.get(tool.function.name):
      print('Calling function:', tool.function.name)
      print('Arguments:', tool.function.arguments)
      # if the function is in the list of asynchronous functions it is executed with asyncio.run()
      if tool.function.name in async_available_functions:
        output = asyncio.run(function_to_call(**tool.function.arguments))
      else:
        output = function_to_call(**tool.function.arguments)
      print('Function output:', output)
    else:
      print('Function', tool.function.name, 'not found')

# Only needed to chat with the model using the tool call results
if response.message.tool_calls:
  # Add the function response to messages for the model to use
  messages.append(response.message)
  messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})

  # Get final response from model with function outputs
  final_response = chat('llama3.1', messages=messages)
  print('Final response:', final_response.message.content)

else:
  print('No tool calls returned from model')