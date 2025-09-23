# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich",
#     "ollama",
# ]
# ///
import os
from typing import Union

from rich import print

from ollama import Client, WebFetchResponse, WebSearchResponse


def format_tool_results(
  results: Union[WebSearchResponse, WebFetchResponse],
  user_search: str,
):
  if isinstance(results, WebSearchResponse):
    output = []
    output.append(f'Search results for "{user_search}":')
    for i, result in enumerate(results.results, 1):
      title = getattr(result, 'title', None)
      url_value = getattr(result, 'url', None)
      output.append(f'{i}. {title}' if title else f'{i}. {getattr(result, "content", "")}')
      if url_value:
        output.append(f'   URL: {url_value}')
      output.append(f'   Content: {getattr(result, "content", "")}')
      output.append('')
    return '\n'.join(output).rstrip()

  elif isinstance(results, WebFetchResponse):
    output = []
    output.append(f'Fetch results for "{user_search}":')
    output.extend([
      f'Title: {results.title}',
      f'URL: {user_search}' if user_search else '',
      f'Content: {results.content}',
    ])
    if results.links:
      output.append(f'Links: {", ".join(results.links)}')
    output.append('')
    return '\n'.join(output).rstrip()

api_key = os.getenv('OLLAMA_API_KEY')
client = Client(headers={'Authorization': f"Bearer {s.getenv('OLLAMA_API_KEY')}"} if api_key else None)
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
        args = tool_call.function.arguments
        result: Union[WebSearchResponse, WebFetchResponse] = function_to_call(**args)
        print('Result from tool call name:', tool_call.function.name, 'with arguments:')
        print(args)
        print()

        user_search = args.get('query', '') or args.get('url', '')
        if tool_call.function.name == 'web_search':
          formatted_tool_results = format_tool_results(result, user_search=user_search)
        elif tool_call.function.name == 'web_fetch':
          formatted_tool_results = format_tool_results(result, user_search=user_search)
        else:
          formatted_tool_results = format_tool_results(result)

        print('Result:')
        print(formatted_tool_results[:200])
        print()

        # caps the result at ~2000 tokens
        messages.append({'role': 'tool', 'content': formatted_tool_results[: 2000 * 4], 'tool_name': tool_call.function.name})
      else:
        print(f'Tool {tool_call.function.name} not found')
        messages.append({'role': 'tool', 'content': f'Tool {tool_call.function.name} not found', 'tool_name': tool_call.function.name})
  else:
    # no more tool calls, we can stop the loop
    break
