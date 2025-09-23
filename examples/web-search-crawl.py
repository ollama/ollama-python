# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich",
#     "ollama",
# ]
# ///
import os
from typing import Optional, Union

from rich import print

from ollama import Client, WebFetchResponse, WebSearchResponse


def format_tool_results(
  results: Union[WebSearchResponse, WebFetchResponse],
  *,
  query: Optional[str] = None,
  url: Optional[str] = None,
):
  if isinstance(results, WebSearchResponse):
    output = []
    if isinstance(results.results, dict):
      for q, search_results in results.results.items():
        output.append(f'Search results for "{q}":')
        for i, result in enumerate(search_results, 1):
          title = getattr(result, 'title', None)
          url_value = getattr(result, 'url', None)
          output.append(f'{i}. {title}' if title else f'{i}. {getattr(result, "content", "")}')
          if url_value:
            output.append(f'   URL: {url_value}')
          output.append(f'   Content: {getattr(result, "content", "")}')
          output.append('')
    else:
      if query:
        output.append(f'Search results for "{query}":')
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
    if url:
      output.append(f'Fetch results for "{url}":')
    output.extend([
      f'Title: {results.title}',
      f'URL: {url}' if url else '',
      f'Content: {results.content}',
    ])
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
        args = tool_call.function.arguments
        result: Union[WebSearchResponse, WebFetchResponse] = function_to_call(**args)
        print('Result from tool call name: ', tool_call.function.name, 'with arguments: ', args)

        if tool_call.function.name == 'web_search':
          formatted = format_tool_results(result, query=args.get('query'))
        elif tool_call.function.name == 'web_fetch':
          formatted = format_tool_results(result, url=args.get('url'))
        else:
          formatted = format_tool_results(result)

        print('Result: ', formatted[:200])

        # caps the result at ~2000 tokens
        messages.append({'role': 'tool', 'content': formatted[: 2000 * 4], 'tool_name': tool_call.function.name})
      else:
        print(f'Tool {tool_call.function.name} not found')
        messages.append({'role': 'tool', 'content': f'Tool {tool_call.function.name} not found', 'tool_name': tool_call.function.name})
  else:
    # no more tool calls, we can stop the loop
    break
