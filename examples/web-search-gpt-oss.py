# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ollama",
# ]
# ///
import os
from typing import Any, Dict, List

from web_search_gpt_oss_helper import Browser

from ollama import Client


def main() -> None:
  if not os.environ.get('OLLAMA_API_KEY'):
    raise SystemExit('Set OLLAMA_API_KEY to an API key from https://ollama.com/settings/keys')

  client = Client()
  browser = Browser(initial_state=None, client=client)

  def browser_search(query: str, topn: int = 10) -> str:
    return browser.search(query=query, topn=topn)['pageText']

  def browser_open(id: int | str | None = None, cursor: int = -1, loc: int = -1, num_lines: int = -1) -> str:
    return browser.open(id=id, cursor=cursor, loc=loc, num_lines=num_lines)['pageText']

  def browser_find(pattern: str, cursor: int = -1, **_: Any) -> str:
    return browser.find(pattern=pattern, cursor=cursor)['pageText']

  browser_search_schema = {
    'type': 'function',
    'function': {
      'name': 'browser.search',
      'description': 'Search the web for a query.',
      'parameters': {
        'type': 'object',
        'properties': {
          'query': {'type': 'string'},
          'topn': {'type': 'integer', 'description': 'Maximum results, from 1 to 10.'},
        },
        'required': ['query'],
      },
    },
  }

  browser_open_schema = {
    'type': 'function',
    'function': {
      'name': 'browser.open',
      'description': 'Open a URL or numbered link, or scroll a page.',
      'parameters': {
        'type': 'object',
        'properties': {
          'id': {'anyOf': [{'type': 'integer'}, {'type': 'string'}]},
          'cursor': {'type': 'integer', 'description': 'Page index; -1 uses the current page.'},
          'loc': {'type': 'integer', 'description': 'First line to display.'},
          'num_lines': {'type': 'integer', 'description': 'Number of lines; -1 uses the default.'},
        },
      },
    },
  }

  browser_find_schema = {
    'type': 'function',
    'function': {
      'name': 'browser.find',
      'description': 'Find text in the current page or a page at a given cursor.',
      'parameters': {
        'type': 'object',
        'properties': {'pattern': {'type': 'string'}, 'cursor': {'type': 'integer'}},
        'required': ['pattern'],
      },
    },
  }

  available_tools = {
    'browser.search': browser_search,
    'browser.open': browser_open,
    'browser.find': browser_find,
  }

  query = "what is ollama's new engine"
  print('Prompt:', query, '\n')

  messages: List[Dict[str, Any]] = [{'role': 'user', 'content': query}]

  while True:
    resp = client.chat(
      model='gpt-oss:120b-cloud',
      messages=messages,
      tools=[browser_search_schema, browser_open_schema, browser_find_schema],
      think=True,
    )

    if resp.message.thinking:
      print('Thinking:\n========\n')
      print(resp.message.thinking + '\n')

    if resp.message.content:
      print('Response:\n========\n')
      print(resp.message.content + '\n')

    messages.append(resp.message)

    if not resp.message.tool_calls:
      break

    for tc in resp.message.tool_calls:
      tool_name = tc.function.name
      args = tc.function.arguments or {}
      print(f'Tool name: {tool_name}, args: {args}')
      fn = available_tools.get(tool_name)
      if not fn:
        messages.append({'role': 'tool', 'content': f'Tool {tool_name} not found', 'tool_name': tool_name})
        continue

      try:
        result_text = fn(**args)
        print('Result: ', result_text[:200] + '...')
      except Exception as e:
        result_text = f'Error from {tool_name}: {e}'

      messages.append({'role': 'tool', 'content': result_text, 'tool_name': tool_name})


if __name__ == '__main__':
  main()
