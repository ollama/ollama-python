import random
from typing import Iterator

from ollama import ChatResponse, Client


def get_temperature(city: str) -> str:
  """
  Get the temperature for a city in Celsius

  Args:
    city (str): The name of the city

  Returns:
    str: The current temperature in Celsius
  """
  # This is a mock implementation - would need to use a real weather API
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
while True:
  response: Iterator[ChatResponse] = client.chat(model, stream=True, messages=messages, tools=[get_temperature, get_conditions], think=True)
  thinking = ''
  content = ''
  tool_calls = []
  for chunk in response:
    if chunk.message.thinking:
      thinking += chunk.message.thinking
      print(chunk.message.thinking, end='', flush=True)
    if chunk.message.content:
      content += chunk.message.content
      print(chunk.message.content, end='', flush=True)
    if chunk.message.tool_calls:
      tool_calls.extend(chunk.message.tool_calls)

  # Preserve the complete assistant turn before adding any tool results.
  messages.append({'role': 'assistant', 'thinking': thinking, 'content': content, 'tool_calls': tool_calls})
  if not tool_calls:
    print()
    break

  for tool in tool_calls:
    function_to_call = available_functions.get(tool.function.name)
    if function_to_call:
      print('\nCalling function:', tool.function.name, 'with arguments:', tool.function.arguments)
      output = function_to_call(**tool.function.arguments)
      print('> Function output:', output)
    else:
      output = f'Function {tool.function.name} not found'
    messages.append({'role': 'tool', 'content': str(output), 'tool_name': tool.function.name})
