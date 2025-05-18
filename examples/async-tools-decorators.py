import asyncio
import ollama
from ollama import ChatResponse
from ollama import ollama_tool, ollama_async_tool, get_tools, get_name_async_tools, get_tools_name


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

available_functions = get_tools_name() # this is a dictionary of tools

# tools are treated differently in synchronous code
async_available_functions = get_name_async_tools()

messages = [{'role': 'user', 'content': 'What is three plus one? and Search the web for what is ollama'}]
print('Prompt:', messages[0]['content'])

async def main():
  client = ollama.AsyncClient()

  response: ChatResponse = await client.chat(
    'llama3.1',
    messages=messages,
    tools=get_tools(),
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
          output = await function_to_call(**tool.function.arguments)
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
    final_response = await client.chat('llama3.1', messages=messages)
    print('Final response:', final_response.message.content)

  else:
    print('No tool calls returned from model')


if __name__ == '__main__':
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print('\nGoodbye!')
