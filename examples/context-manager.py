# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "ollama",
# ]
# ///
"""
Example demonstrating how to use Client and AsyncClient as context managers
to ensure proper cleanup and release of underlying HTTP connections.
"""

import asyncio

from ollama import AsyncClient, Client


def sync_example():
  print('=== Synchronous Client Context Manager ===')
  # Entering the context manager initializes the client;
  # exiting cleanly closes all httpx connection pools.
  with Client() as client:
    models = client.list()
    print(f'Discovered {len(models.models)} installed models:')
    for model in models.models[:3]:
      print(f' - {model.model} (modified: {model.modified_at})')


async def async_example():
  print('\n=== Asynchronous Client Context Manager ===')
  async with AsyncClient() as client:
    response = await client.generate(
      model='gemma4',
      prompt='Why is the sky blue in one sentence?',
    )
    print(f'Response: {response.response}')


if __name__ == '__main__':
  sync_example()
  # Run async example if an async event loop is available
  try:
    asyncio.run(async_example())
  except Exception as e:
    print(f'Async example skipped or encountered error: {e}')
