import ollama


def print_logprobs(logprobs, label: str) -> None:
  print(f'\n{label}:')
  for entry in logprobs:
    token = entry.get('token', '')
    logprob = entry.get('logprob')
    print(f'  token={token!r:<12} logprob={logprob:.3f}')
    for alt in entry.top_logprobs or []:
      if alt['token'] != token:
        print(f'    alt -> {alt["token"]!r:<12} ({alt["logprob"]:.3f})')


response = ollama.generate(
  model='gemma4',
  prompt='hi! be concise.',
  logprobs=True,
  top_logprobs=3,
)
print('Generate response:', response['response'])
print_logprobs(response.logprobs or [], 'generate logprobs')
