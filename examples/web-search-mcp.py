# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp",
#   "rich",
#   "ollama",
# ]
# ///
"""
MCP stdio server exposing Ollama web_search and web_fetch as tools.

Environment:
- OLLAMA_API_KEY (required): if set, will be used as Authorization header.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from ollama import Client

try:
  # Preferred high-level API (if available)
  from mcp.server.fastmcp import FastMCP  # type: ignore

  _FASTMCP_AVAILABLE = True
except Exception:
  _FASTMCP_AVAILABLE = False

if not _FASTMCP_AVAILABLE:
  # Fallback to the low-level stdio server API
  from mcp.server import Server  # type: ignore
  from mcp.server.stdio import stdio_server  # type: ignore


client = Client()


def _web_search_impl(query: str, max_results: int = 3) -> Dict[str, Any]:
  res = client.web_search(query=query, max_results=max_results)
  return res.model_dump()


def _web_fetch_impl(url: str) -> Dict[str, Any]:
  res = client.web_fetch(url=url)
  return res.model_dump()


if _FASTMCP_AVAILABLE:
  app = FastMCP('ollama-search-fetch')

  @app.tool()
  def web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Perform a web search using Ollama's hosted search API.

    Args:
      query: The search query to run.
      max_results: Maximum results to return (default: 3).

    Returns:
      JSON-serializable dict matching ollama.WebSearchResponse.model_dump()
    """

    return _web_search_impl(query=query, max_results=max_results)

  @app.tool()
  def web_fetch(url: str) -> Dict[str, Any]:
    """
    Fetch the content of a web page for the provided URL.

    Args:
      url: The absolute URL to fetch.

    Returns:
      JSON-serializable dict matching ollama.WebFetchResponse.model_dump()
    """

    return _web_fetch_impl(url=url)

  if __name__ == '__main__':
    app.run()

else:
  server = Server('ollama-search-fetch')  # type: ignore[name-defined]

  @server.tool()  # type: ignore[attr-defined]
  async def web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Perform a web search using Ollama's hosted search API.

    Args:
      query: The search query to run.
      max_results: Maximum results to return (default: 3).
    """

    return await asyncio.to_thread(_web_search_impl, query, max_results)

  @server.tool()  # type: ignore[attr-defined]
  async def web_fetch(url: str) -> Dict[str, Any]:
    """
    Fetch the content of a web page for the provided URL.

    Args:
      url: The absolute URL to fetch.
    """

    return await asyncio.to_thread(_web_fetch_impl, url)

  async def _main() -> None:
    async with stdio_server() as (read, write):  # type: ignore[name-defined]
      await server.run(read, write)  # type: ignore[attr-defined]

  if __name__ == '__main__':
    asyncio.run(_main())
