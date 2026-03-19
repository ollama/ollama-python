# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cross-validated-search",
#     "ollama",
#     "rich",
# ]
# ///
"""
Cross-Validated Search Example for Ollama

This example demonstrates how to use cross-validated-search to prevent
hallucinations in LLM responses by verifying facts against multiple sources.

cross-validated-search provides:
- Confidence scoring (Verified/Likely True/Uncertain/Likely False)
- Cross-validation across multiple search engines
- No API key required - completely free

Install: pip install cross-validated-search
"""

from rich import print
from rich.table import Table
from rich.panel import Panel

from ollama import chat
from cross_validated_search import CrossValidatedSearcher


def search_with_confidence(query: str) -> dict:
  """
  Perform a cross-validated web search and return results with confidence scores.
  """
  searcher = CrossValidatedSearcher()
  results = searcher.search(query, search_type='text', max_results=5)

  return {
    'query': query,
    'answer': results.answer,
    'confidence': results.confidence,
    'sources': [
      {
        'engine': r.engine,
        'title': r.title,
        'url': r.url,
        'snippet': r.snippet[:100] + '...' if len(r.snippet) > 100 else r.snippet,
      }
      for r in results.sources
    ],
  }


def format_context_for_llm(search_results: dict) -> str:
  """
  Format search results as context for the LLM.
  """
  context_parts = [f"Query: {search_results['query']}"]

  # Add confidence indicator
  confidence_emoji = {
    'verified': '✅',
    'likely_true': '🟢',
    'uncertain': '🟡',
    'likely_false': '🔴',
  }
  emoji = confidence_emoji.get(search_results['confidence'], '❓')
  context_parts.append(f"Confidence: {emoji} {search_results['confidence']}")

  # Add sources
  context_parts.append("\nSources:")
  for i, source in enumerate(search_results['sources'][:3], 1):
    context_parts.append(f"{i}. [{source['engine']}] {source['title']}")
    context_parts.append(f"   {source['url']}")

  # Add answer summary
  if search_results['answer']:
    context_parts.append(f"\nSummary: {search_results['answer']}")

  return '\n'.join(context_parts)


def main():
  print(Panel.fit(
    "[bold blue]Cross-Validated Search + Ollama[/bold blue]\n"
    "Prevent hallucinations with confidence-scored search results",
    title="🥚 cross-validated-search",
  ))

  # Example queries to test
  queries = [
    "What is the latest version of Python?",
    "When was the first iPhone released?",
    "What is the population of Tokyo?",
  ]

  for query in queries:
    print(f"\n[bold cyan]Searching:[/bold cyan] {query}")

    # Step 1: Perform cross-validated search
    search_results = search_with_confidence(query)

    # Step 2: Display search results with confidence
    table = Table(title=f"Results for: {query}")
    table.add_column("Engine", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Confidence", style="bold")

    for source in search_results['sources'][:3]:
      table.add_row(source['engine'], source['title'][:40], search_results['confidence'])

    print(table)

    # Step 3: Format context for LLM
    context = format_context_for_llm(search_results)

    # Step 4: Query LLM with verified context
    print(f"\n[bold yellow]Asking LLM with verified context...[/bold yellow]")

    response = chat(
      model='llama3.2',
      messages=[
        {
          'role': 'system',
          'content': 'You are a helpful assistant. Always cite your sources and indicate confidence levels when answering factual questions. If the confidence is low, acknowledge uncertainty.',
        },
        {
          'role': 'user',
          'content': f"Based on these verified search results, please answer:\n\n{context}\n\nPlease provide a concise answer and cite the sources.",
        },
      ],
    )

    print(f"\n[bold green]LLM Response:[/bold green]")
    print(response.message.content)

    print("\n" + "=" * 60)


if __name__ == '__main__':
  main()