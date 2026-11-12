from datetime import datetime

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ollama import ChatResponse, chat


def get_current_time(timezone: str = 'UTC') -> str:
  """
  Get the current date and time.

  Args:
    timezone (str): IANA timezone name, for example 'Asia/Kolkata' or 'UTC'

  Returns:
    str: A sentence stating the current date and time in the given timezone
  """
  try:
    now = datetime.now(ZoneInfo(timezone))
  except ZoneInfoNotFoundError:
    return f'Unknown timezone: {timezone}'
  # Return a full sentence so smaller models use it directly in their answer
  return f'The current date and time in {timezone} is ' + now.strftime('%A, %d %B %Y, %I:%M %p %Z')


available_functions = {
  'get_current_time': get_current_time,
}

messages = [{'role': 'user', 'content': 'What time is it in Tokyo right now?'}]
print('Prompt:', messages[0]['content'])

response: ChatResponse = chat(
  'llama3.1',
  messages=messages,
  tools=[get_current_time],  # the function is converted to a tool schema automatically
)

if response.message.tool_calls:
  # Keep the assistant's tool call in the history
  messages.append(response.message)

  for tool in response.message.tool_calls:
    if function_to_call := available_functions.get(tool.function.name):
      print('Calling function:', tool.function.name)
      print('Arguments:', tool.function.arguments)
      output = function_to_call(**tool.function.arguments)
      print('Function output:', output)
      messages.append({'role': 'tool', 'tool_name': tool.function.name, 'content': str(output)})
    else:
      print('Function', tool.function.name, 'not found')

  # Send the tool result back so the model can answer in natural language
  final_response = chat('llama3.1', messages=messages, tools=[get_current_time])
  print('Final response:', final_response.message.content)
else:
  print('No tool calls returned from model')
  print(response.message.content)
