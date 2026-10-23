# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich",
#     "ollama",
# ]
# ///
import os
import sys

from rich import print

from ollama import Client

api_key = os.getenv('OLLAMA_API_KEY')
if not api_key:
  print('[bold red]Error:[/bold red] OLLAMA_API_KEY environment variable is required.')
  print('Get your API key at https://ollama.com/settings/keys')
  sys.exit(1)

# You can pass api_key directly or rely on OLLAMA_API_KEY
client = Client(api_key=api_key)

url = 'https://ollama.com'
print(f'Fetching content from: [bold cyan]{url}[/bold cyan]\n')

# Fetch using the client
response = client.web_fetch(url)

print(f'Title: [bold green]{response.title}[/bold green]')
if response.links:
  print(f'Found {len(response.links)} links on the page:')
  for link in response.links[:5]:
    print(f'  - {link}')

print('\nContent Preview:')
print(response.content[:500] if response.content else 'No content returned')
