import sys

from ollama import create


args = sys.argv[1:]
if len(args) == 2:
  # create from local file
  path = args[1]
else:
  print('usage: python main.py <name> <filepath>')
  sys.exit(1)

# TODO: update to real Modelfile values
modelfile = f"""
FROM {path}
"""

for response in create(model=args[0], modelfile=modelfile, stream=True):
  print(response['status'])
