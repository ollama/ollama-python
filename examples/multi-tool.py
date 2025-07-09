import random
from typing import Iterator

from ollama import ChatResponse, Client


def get_temperature(city: str) -> int:
  """
  Get the temperature for a city in Celsius

  Args:
    city (str): The name of the city

  Returns:
    int: The current temperature in Celsius
  """
  # This is a mock implementation - would need to use a real weather API
  import random

  if city not in ['London', 'Paris', 'New York', 'Tokyo', 'Sydney']:
    return 'Unknown city'

  return str(random.randint(0, 35)) + ' degrees Celsius'


def get_conditions(city: str) -> str:
  """
  Get the weather conditions for a city
  """
  if city not in ['London', 'Paris', 'New York', 'Tokyo', 'Sydney']:
    return 'Unknown city'
  # This is a mock implementation - would need to use a real weather API
  conditions = ['sunny', 'cloudy', 'rainy', 'snowy']
  return random.choice(conditions)


available_functions = {
  'get_temperature': get_temperature,
  'get_conditions': get_conditions,
}


cities = ['London', 'Paris', 'New York', 'Tokyo', 'Sydney']
city = random.choice(cities)
city2 = random.choice(cities)
messages = [{'role': 'user', 'content': f'What is the temperature in {city}? and what are the weather conditions in {city2}?'}]
print('----- Prompt:', messages[0]['content'], '\n')

model = 'qwen3'
client = Client()
response: Iterator[ChatResponse] = client.chat(model, stream=True, messages=messages, tools=[get_temperature, get_conditions], think=True)

for chunk in response:
  if chunk.message.thinking:
    print(chunk.message.thinking, end='', flush=True)
  if chunk.message.content:
    print(chunk.message.content, end='', flush=True)
  if chunk.message.tool_calls:
    for tool in chunk.message.tool_calls:
      if function_to_call := available_functions.get(tool.function.name):
        print('\nCalling function:', tool.function.name, 'with arguments:', tool.function.arguments)
        output = function_to_call(**tool.function.arguments)
        print('> Function output:', output, '\n')

        # Add the assistant message and tool call result to the messages
        messages.append(chunk.message)
        messages.append({'role': 'tool', 'content': str(output), 'tool_name': tool.function.name})
      else:
        print('Function', tool.function.name, 'not found')

print('----- Sending result back to model \n')
if any(msg.get('role') == 'tool' for msg in messages):
  res = client.chat(model, stream=True, tools=[get_temperature, get_conditions], messages=messages, think=True)
  done_thinking = False
  for chunk in res:
    if chunk.message.thinking:
      print(chunk.message.thinking, end='', flush=True)
    if chunk.message.content:
      if not done_thinking:
        print('\n----- Final result:')
        done_thinking = True
      print(chunk.message.content, end='', flush=True)
    if chunk.message.tool_calls:
      # Model should be explaining the tool calls and the results in this output
      print('Model returned tool calls:')
      print(chunk.message.tool_calls)
else:
  print('No tool calls returned')
