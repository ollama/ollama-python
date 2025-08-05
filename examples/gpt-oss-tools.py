import random

from ollama import chat
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


model = 'gpt-oss:20b'
# gpt-oss can call tools while "thinking"
# a loop is needed to call the tools and get the results
while True:
  response: ChatResponse = chat(model=model, messages=messages, tools=[get_weather, get_weather_conditions])

  if response.message.content:
    print('Content: ')
    print(response.message.content + '\n')
  if response.message.thinking:
    print('Thinking: ')
    print(response.message.thinking + '\n')

  if response.message.tool_calls:
    for tool_call in response.message.tool_calls:
      function_to_call = available_tools.get(tool_call.function.name)
      if function_to_call:
        result = function_to_call(**tool_call.function.arguments)
        print('Result from tool call name: ', tool_call.function.name, 'with arguments: ', tool_call.function.arguments, 'result: ', result + '\n')

        messages.append(response.message)
        messages.append({'role': 'tool', 'content': result, 'tool_name': tool_call.function.name})
      else:
        print(f'Tool {tool_call.function.name} not found')

  else:
    # no more tool calls, we can stop the loop
    break
