# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp>=2,<3",
#   "ollama>=0.6.3",
# ]
# ///
"""MCP stdio server exposing Ollama web search and fetch. Requires OLLAMA_API_KEY."""

import os

from mcp.server.mcpserver import MCPServer

from ollama import Client

if not os.environ.get('OLLAMA_API_KEY'):
  raise SystemExit('Set OLLAMA_API_KEY to an API key from https://ollama.com/settings/keys')

app = MCPServer('ollama-search-fetch')
client = Client()


@app.tool()
def web_search(query: str, max_results: int = 3) -> dict:
  """Search the web. max_results must be between 1 and 10."""
  return client.web_search(query=query, max_results=max_results).model_dump()


@app.tool()
def web_fetch(url: str) -> dict:
  """Fetch the content of a web page at an absolute URL."""
  return client.web_fetch(url=url).model_dump()


if __name__ == '__main__':
  try:
    app.run()
  finally:
    client.close()
