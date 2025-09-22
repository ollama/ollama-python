# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich",
#     "ollama",
# ]
# ///
from typing import Union

from ollama import Client, WebFetchResponse, WebSearchResponse


def format_tool_results(results: Union[WebSearchResponse, WebFetchResponse]):
  if isinstance(results, WebSearchResponse):
    output = []
    for i, result in enumerate(results.results, 1):
      output.append(f'{i}. {result.content}')
      output.append('')

    return '\n'.join(output).rstrip()

  elif isinstance(results, WebFetchResponse):
    output = [
      f'Title: {results.title}',
      f'Content: {results.content}',
    ]
    if results.links:
      output.append(f'Links: {", ".join(results.links)}')
    output.append('')

    return '\n'.join(output).rstrip()


client = Client(headers={'Authorization': (os.getenv('OLLAMA_API_KEY'))})
available_tools = {'web_search': client.web_search, 'web_fetch': client.web_fetch}

query = "ollama's new engine"
print('Query: ', query)

messages = [{'role': 'user', 'content': query}]
while True:
  response = client.chat(model='qwen3', messages=messages, tools=[client.web_search, client.web_fetch], think=True)
  if response.message.thinking:
    print('Thinking: ')
    print(response.message.thinking + '\n\n')
  if response.message.content:
    print('Content: ')
    print(response.message.content + '\n')

  messages.append(response.message)

  if response.message.tool_calls:
    for tool_call in response.message.tool_calls:
      function_to_call = available_tools.get(tool_call.function.name)
      if function_to_call:
        result: Union[WebSearchResponse, WebFetchResponse] = function_to_call(**tool_call.function.arguments)
        print('Result from tool call name: ', tool_call.function.name, 'with arguments: ', tool_call.function.arguments)
        print('Result: ', format_tool_results(result)[:200])

        # caps the result at ~2000 tokens
        messages.append({'role': 'tool', 'content': format_tool_results(result)[: 2000 * 4], 'tool_name': tool_call.function.name})
      else:
        print(f'Tool {tool_call.function.name} not found')
        messages.append({'role': 'tool', 'content': f'Tool {tool_call.function.name} not found', 'tool_name': tool_call.function.name})
  else:
    # no more tool calls, we can stop the loop
    break
