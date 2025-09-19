from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from ollama import Client

try:
  from .browser_tool_helpers import Browser  # when run with -m
except Exception:
  from browser_tool_helpers import Browser  # when run as a script

def main() -> None:
  client = Client(headers={'Authorization': os.getenv('OLLAMA_API_KEY')})
  browser = Browser(initial_state=None, client=client)

  # Minimal tool schemas (match other examples: names only)
  browser_search_schema = {'type': 'function', 'function': {'name': 'browser.search'}}
  browser_open_schema = {'type': 'function', 'function': {'name': 'browser.open'}}
  browser_find_schema = {'type': 'function', 'function': {'name': 'browser.find'}}

  # Simple wrappers returning page text
  def browser_search(query: str, topn: int = 10) -> str:
    return browser.search(query=query, topn=topn)['pageText']

  def browser_open(id: int | str = -1, cursor: int = -1, loc: int = -1, num_lines: int = -1) -> str:
    return browser.open(id=id, cursor=cursor, loc=loc, num_lines=num_lines)['pageText']

  def browser_find(pattern: str, cursor: int = -1) -> str:
    return browser.find(pattern=pattern, cursor=cursor)['pageText']

  available_tools = {
    'browser.search': browser_search,
    'browser.open': browser_open,
    'browser.find': browser_find,
  }

  messages: List[Dict[str, Any]] = [{'role': 'user', 'content': 'What is Ollama?'}]
  print('----- Prompt:', messages[0]['content'], '\n')

  while True:
    resp = client.chat(
      model='gpt-oss',
      messages=messages,
      tools=[browser_search_schema, browser_open_schema, browser_find_schema],
      think=True,
    )

    if getattr(resp.message, 'thinking', None):
      print('Thinking:\n========\n')
      print(resp.message.thinking + '\n')

    if getattr(resp.message, 'content', None):
      print('Response:\n========\n')
      print(resp.message.content + '\n')

    messages.append(resp.message)

    if not resp.message.tool_calls:
      break

    for tc in resp.message.tool_calls:
      tool_name = tc.function.name
      args = tc.function.arguments or {}
      fn = available_tools.get(tool_name)
      if not fn:
        messages.append({'role': 'tool', 'content': f'Tool {tool_name} not found', 'tool_name': tool_name})
        continue

      try:
        result_text = fn(**args)
      except Exception as e:
        result_text = f'Error from {tool_name}: {e}'

      messages.append({'role': 'tool', 'content': result_text, 'tool_name': tool_name})


if __name__ == '__main__':
  main()
