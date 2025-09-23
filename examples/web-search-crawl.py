# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich",
#     "ollama",
# ]
# ///
from typing import Union

from rich import print

from ollama import WebFetchResponse, WebSearchResponse, chat, web_fetch, web_search


def format_tool_results(
  results: Union[WebSearchResponse, WebFetchResponse],
  user_search: str,
):
  output = []
  if isinstance(results, WebSearchResponse):
    output.append(f'Search results for "{user_search}":')
    for result in results.results:
      output.append(f'{result.title}' if result.title else f'{result.content}')
      output.append(f'   URL: {result.url}')
      output.append(f'   Content: {result.content}')
      output.append('')
    return '\n'.join(output).rstrip()

  elif isinstance(results, WebFetchResponse):
    output.append(f'Fetch results for "{user_search}":')
    output.extend(
      [
        f'Title: {results.title}',
        f'URL: {user_search}' if user_search else '',
        f'Content: {results.content}',
      ]
    )
    if results.links:
      output.append(f'Links: {", ".join(results.links)}')
    output.append('')
    return '\n'.join(output).rstrip()


# client = Client(headers={'Authorization': f"Bearer {os.getenv('OLLAMA_API_KEY')}"} if api_key else None)
available_tools = {'web_search': web_search, 'web_fetch': web_fetch}

query = "what is ollama's new engine"
print('Query: ', query)

messages = [{'role': 'user', 'content': query}]
while True:
  response = chat(model='qwen3', messages=messages, tools=[web_search, web_fetch], think=True)
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
        formatted_tool_results = format_tool_results(result, user_search=user_search)

        print(formatted_tool_results[:300])
        print()

        # caps the result at ~2000 tokens
        messages.append({'role': 'tool', 'content': formatted_tool_results[: 2000 * 4], 'tool_name': tool_call.function.name})
      else:
        print(f'Tool {tool_call.function.name} not found')
        messages.append({'role': 'tool', 'content': f'Tool {tool_call.function.name} not found', 'tool_name': tool_call.function.name})
  else:
    # no more tool calls, we can stop the loop
    break
