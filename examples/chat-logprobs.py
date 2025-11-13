from typing import Iterable

import ollama


def print_logprobs(logprobs: Iterable[dict], label: str) -> None:
  print(f'\n{label}:')
  for entry in logprobs:
    token = entry.get('token', '')
    logprob = entry.get('logprob')
    print(f'  token={token!r:<12} logprob={logprob:.3f}')
    for alt in entry.get('top_logprobs', []):
      if alt['token'] != token:
        print(f'    alt -> {alt["token"]!r:<12} ({alt["logprob"]:.3f})')


messages = [
  {
    'role': 'user',
    'content': 'hi! be concise.',
  },
]

response = ollama.chat(
  model='gemma3',
  messages=messages,
  logprobs=True,
  top_logprobs=3,
)
print('Chat response:', response['message']['content'])
print_logprobs(response.get('logprobs', []), 'chat logprobs')
