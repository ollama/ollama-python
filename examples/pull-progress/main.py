from tqdm import tqdm
from ollama import pull


current_digest, bars = '', {}
for progress in pull('mistral', stream=True):
  digest = progress.get('digest', '')
  if digest != current_digest and current_digest in bars:
    bars[current_digest].close()

  if not digest:
    print(progress.get('status'))
    continue

  if digest not in bars and (total := progress.get('total')):
    bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)

  if completed := progress.get('completed'):
    bars[digest].update(completed - bars[digest].n)

  current_digest = digest
