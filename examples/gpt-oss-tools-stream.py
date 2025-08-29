# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gpt-oss",
#     "ollama",
#     "rich",
# ]
# ///
import random
from typing import Iterator

from rich import print

from ollama import Client
from ollama._types import ChatResponse


def get_weather(city: str) -> str:
  """
  Get the current temperature for a city

  Args:
      city (str): The name of the city

  Returns:
      str: The current temperature
  """
  temperatures = list(range(-10, 35))

  temp = random.choice(temperatures)

  return f'The temperature in {city} is {temp}°C'


def get_weather_conditions(city: str) -> str:
  """
  Get the weather conditions for a city

  Args:
      city (str): The name of the city

  Returns:
      str: The current weather conditions
  """
  conditions = ['sunny', 'cloudy', 'rainy', 'snowy', 'foggy']
  return random.choice(conditions)


available_tools = {'get_weather': get_weather, 'get_weather_conditions': get_weather_conditions}

messages = [{'role': 'user', 'content': 'What is the weather like in London? What are the conditions in Toronto?'}]

client = Client(
  # Ollama Turbo
  # host="https://ollama.com", headers={'Authorization': (os.getenv('OLLAMA_API_KEY'))}
)

model = 'gpt-oss:20b'
# gpt-oss can call tools while "thinking"
# a loop is needed to call the tools and get the results
final = True
while True:
  response_stream: Iterator[ChatResponse] = client.chat(model=model, messages=messages, tools=[get_weather, get_weather_conditions], stream=True)
  tool_calls = []
  thinking = ''
  content = ''

  for chunk in response_stream:
    if chunk.message.tool_calls:
      tool_calls.extend(chunk.message.tool_calls)

    if chunk.message.content:
      if not (chunk.message.thinking or chunk.message.thinking == '') and final:
        print('\n\n' + '=' * 10)
        print('Final result: ')
        final = False
      print(chunk.message.content, end='', flush=True)

    if chunk.message.thinking:
      # accumulate thinking
      thinking += chunk.message.thinking
      print(chunk.message.thinking, end='', flush=True)

  if thinking != '' or content != '' or len(tool_calls) > 0:
    messages.append({'role': 'assistant', 'thinking': thinking, 'content': content, 'tool_calls': tool_calls})

  print()

  if tool_calls:
    for tool_call in tool_calls:
      function_to_call = available_tools.get(tool_call.function.name)
      if function_to_call:
        print('\nCalling tool:', tool_call.function.name, 'with arguments: ', tool_call.function.arguments)
        result = function_to_call(**tool_call.function.arguments)
        print('Tool result: ', result + '\n')

        result_message = {'role': 'tool', 'content': result, 'tool_name': tool_call.function.name}
        messages.append(result_message)
      else:
        print(f'Tool {tool_call.function.name} not found')
        messages.append({'role': 'tool', 'content': f'Tool {tool_call.function.name} not found', 'tool_name': tool_call.function.name})

  else:
    # no more tool calls, we can stop the loop
    break
