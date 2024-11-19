import sys

from ollama import create


args = sys.argv[1:]
if len(args) == 2:
  # create from local file
  path = args[1]
else:
  print('usage: python create.py <name> <filepath>')
  sys.exit(1)

# TODO: update to real Modelfile values
modelfile = f"""
FROM {path}
"""
example_modelfile = """
FROM llama3.2
# sets the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1
# sets the context window size to 4096, this controls how many tokens the LLM can use as context to generate the next token
PARAMETER num_ctx 4096

# sets a custom system message to specify the behavior of the chat assistant
SYSTEM You are Mario from super mario bros, acting as an assistant.
"""

for response in create(model=args[0], modelfile=modelfile, stream=True):
  print(response['status'])
