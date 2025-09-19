# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp",
#   "rich",
#   "ollama",
# ]
# ///
"""
Minimal MCP stdio server exposing Ollama web_search and web_crawl as tools.

This lets MCP clients (e.g., Cursor, Claude Desktop) call these tools.

Environment:
- OLLAMA_API_KEY (optional): if set, will be used as Authorization header.

Run directly (stdio transport):
  python examples/mcp_web_search_crawl_server.py

In Cursor/Claude MCP config, point a command to this script.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

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


def _make_client() -> Client:
  headers = {}
  api_key = os.getenv("OLLAMA_API_KEY")
  if api_key:
    headers["Authorization"] = api_key
  return Client(headers=headers)


client = _make_client()


def _web_search_impl(queries: List[str], max_results: int = 3) -> Dict[str, Any]:
  res = client.web_search(queries=queries, max_results=max_results)
  return res.model_dump()


def _web_crawl_impl(urls: List[str]) -> Dict[str, Any]:
  res = client.web_crawl(urls=urls)
  return res.model_dump()


if _FASTMCP_AVAILABLE:
  app = FastMCP("ollama-web-tools")

  @app.tool()
  def web_search(queries: List[str], max_results: int = 3) -> Dict[str, Any]:
    """
    Perform a web search using Ollama's hosted search API.

    Args:
      queries: A list of search queries to run.
      max_results: Maximum results per query (default: 3).

    Returns:
      JSON-serializable dict matching ollama.WebSearchResponse.model_dump()
    """

    return _web_search_impl(queries=queries, max_results=max_results)

  @app.tool()
  def web_crawl(urls: List[str]) -> Dict[str, Any]:
    """
    Crawl one or more web pages and return extracted content.

    Args:
      urls: A list of absolute URLs to crawl.

    Returns:
      JSON-serializable dict matching ollama.WebCrawlResponse.model_dump()
    """

    return _web_crawl_impl(urls=urls)

  if __name__ == "__main__":
    app.run()

else:
  server = Server("ollama-web-tools")  # type: ignore[name-defined]

  @server.tool()  # type: ignore[attr-defined]
  async def web_search(queries: List[str], max_results: int = 3) -> Dict[str, Any]:
    """
    Perform a web search using Ollama's hosted search API.

    Args:
      queries: A list of search queries to run.
      max_results: Maximum results per query (default: 3).
    """

    return await asyncio.to_thread(_web_search_impl, queries, max_results)

  @server.tool()  # type: ignore[attr-defined]
  async def web_crawl(urls: List[str]) -> Dict[str, Any]:
    """
    Crawl one or more web pages and return extracted content.

    Args:
      urls: A list of absolute URLs to crawl.
    """

    return await asyncio.to_thread(_web_crawl_impl, urls)

  async def _main() -> None:
    async with stdio_server() as (read, write):  # type: ignore[name-defined]
      await server.run(read, write)  # type: ignore[attr-defined]

  if __name__ == "__main__":
    asyncio.run(_main())

