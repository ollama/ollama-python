# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gpt-oss",
#     "ollama",
#     "rich",
# ]
# ///

import asyncio
import json
from typing import Iterator, Optional

from gpt_oss.tools.simple_browser import ExaBackend, SimpleBrowserTool
from openai_harmony import Author, Role, TextContent
from openai_harmony import Message as HarmonyMessage
from rich import print

from ollama import Client
from ollama._types import ChatResponse

_backend = ExaBackend(source='web')
_browser_tool = SimpleBrowserTool(backend=_backend)


def heading(text):
  print(text)
  print('=' * (len(text) + 3))


async def _browser_search_async(query: str, topn: int = 10, source: str | None = None) -> str:
  # map Ollama message to Harmony format
  harmony_message = HarmonyMessage(
    author=Author(role=Role.USER),
    content=[TextContent(text=json.dumps({'query': query, 'topn': topn}))],
    recipient='browser.search',
  )

  result_text: str = ''
  async for response in _browser_tool._process(harmony_message):
    if response.content:
      for content in response.content:
        if isinstance(content, TextContent):
          result_text += content.text
  return result_text or f'No results for query: {query}'


async def _browser_open_async(id: int | str = -1, cursor: int = -1, loc: int = -1, num_lines: int = -1, *, view_source: bool = False, source: str | None = None) -> str:
  payload = {'id': id, 'cursor': cursor, 'loc': loc, 'num_lines': num_lines, 'view_source': view_source, 'source': source}

  harmony_message = HarmonyMessage(
    author=Author(role=Role.USER),
    content=[TextContent(text=json.dumps(payload))],
    recipient='browser.open',
  )

  result_text: str = ''
  async for response in _browser_tool._process(harmony_message):
    if response.content:
      for content in response.content:
        if isinstance(content, TextContent):
          result_text += content.text
  return result_text or f'Could not open: {id}'


async def _browser_find_async(pattern: str, cursor: int = -1) -> str:
  payload = {'pattern': pattern, 'cursor': cursor}

  harmony_message = HarmonyMessage(
    author=Author(role=Role.USER),
    content=[TextContent(text=json.dumps(payload))],
    recipient='browser.find',
  )

  result_text: str = ''
  async for response in _browser_tool._process(harmony_message):
    if response.content:
      for content in response.content:
        if isinstance(content, TextContent):
          result_text += content.text
  return result_text or f'Pattern not found: {pattern}'


def browser_search(query: str, topn: int = 10, source: Optional[str] = None) -> str:
  return asyncio.run(_browser_search_async(query=query, topn=topn, source=source))


def browser_open(id: int | str = -1, cursor: int = -1, loc: int = -1, num_lines: int = -1, *, view_source: bool = False, source: Optional[str] = None) -> str:
  return asyncio.run(_browser_open_async(id=id, cursor=cursor, loc=loc, num_lines=num_lines, view_source=view_source, source=source))


def browser_find(pattern: str, cursor: int = -1) -> str:
  return asyncio.run(_browser_find_async(pattern=pattern, cursor=cursor))


# Schema definitions for each browser tool
browser_search_schema = {
  'type': 'function',
  'function': {
    'name': 'browser.search',
  },
}

browser_open_schema = {
  'type': 'function',
  'function': {
    'name': 'browser.open',
  },
}

browser_find_schema = {
  'type': 'function',
  'function': {
    'name': 'browser.find',
  },
}

available_tools = {
  'browser.search': browser_search,
  'browser.open': browser_open,
  'browser.find': browser_find,
}


model = 'gpt-oss:20b'
print('Model: ', model, '\n')

prompt = 'What is Ollama?'
print('You: ', prompt, '\n')
messages = [{'role': 'user', 'content': prompt}]

client = Client()

# gpt-oss can call tools while "thinking"
# a loop is needed to call the tools and get the results
final = True
while True:
  response_stream: Iterator[ChatResponse] = client.chat(
    model=model,
    messages=messages,
    tools=[browser_search_schema, browser_open_schema, browser_find_schema],
    options={'num_ctx': 8192},  # 8192 is the recommended lower limit for the context window
    stream=True,
  )

  tool_calls = []
  thinking = ''
  content = ''

  for chunk in response_stream:
    if chunk.message.tool_calls:
      tool_calls.extend(chunk.message.tool_calls)

    if chunk.message.content:
      if not (chunk.message.thinking or chunk.message.thinking == '') and final:
        heading('\n\nFinal result: ')
        final = False
      print(chunk.message.content, end='', flush=True)

    if chunk.message.thinking:
      thinking += chunk.message.thinking
      print(chunk.message.thinking, end='', flush=True)

  if thinking != '':
    messages.append({'role': 'assistant', 'content': thinking, 'tool_calls': tool_calls})

  print()

  if tool_calls:
    for tool_call in tool_calls:
      tool_name = tool_call.function.name
      args = tool_call.function.arguments or {}
      function_to_call = available_tools.get(tool_name)

      if function_to_call:
        heading(f'\nCalling tool: {tool_name}')
        if args:
          print(f'Arguments: {args}')

        try:
          result = function_to_call(**args)
          print(f'Tool result: {result[:200]}')
          if len(result) > 200:
            heading('... [truncated]')
          print()

          result_message = {'role': 'tool', 'content': result, 'tool_name': tool_name}
          messages.append(result_message)

        except Exception as e:
          err = f'Error from {tool_name}: {e}'
          print(err)
          messages.append({'role': 'tool', 'content': err, 'tool_name': tool_name})
      else:
        print(f'Tool {tool_name} not found')
  else:
    # no more tool calls, we can stop the loop
    break
