# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich",
# ]
# ///
import os
from typing import Union

from rich import print

from ollama import Client, WebCrawlResponse, WebSearchResponse


def format_tool_results(results: Union[WebSearchResponse, WebCrawlResponse]):
  if isinstance(results, WebSearchResponse):
    if not results.success:
      error_msg = ', '.join(results.errors) if results.errors else 'Unknown error'
      return f'Web search failed: {error_msg}'

    output = []
    for query, search_results in results.results.items():
      output.append(f'Search results for "{query}":')
      for i, result in enumerate(search_results, 1):
        output.append(f'{i}. {result.title}')
        output.append(f'   URL: {result.url}')
        output.append(f'   Content: {result.content}')
        output.append('')

    return '\n'.join(output).rstrip()

  elif isinstance(results, WebCrawlResponse):
    if not results.success:
      error_msg = ', '.join(results.errors) if results.errors else 'Unknown error'
      return f'Web crawl failed: {error_msg}'

    output = []
    for url, crawl_results in results.results.items():
      output.append(f'Crawl results for "{url}":')
      for i, result in enumerate(crawl_results, 1):
        output.append(f'{i}. {result.title}')
        output.append(f'   URL: {result.url}')
        output.append(f'   Content: {result.content}')
        if result.links:
          output.append(f'   Links: {", ".join(result.links)}')
        output.append('')

    return '\n'.join(output).rstrip()


client = Client(headers={'Authorization': (os.getenv('OLLAMA_API_KEY'))})
available_tools = {'websearch': client.websearch, 'webcrawl': client.webcrawl}

query = "ollama's new engine"
print('Query: ', query)

messages = [{'role': 'user', 'content': query}]
while True:
  response = client.chat(model='qwen3', messages=messages, tools=[client.websearch, client.webcrawl], think=True)
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
        result: WebSearchResponse | WebCrawlResponse = function_to_call(**tool_call.function.arguments)
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
