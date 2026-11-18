from ollama import Client

client = Client()

# Each keyword argument maps to a Modelfile directive; together they are the
# Modelfile, expressed in Python:
#   from_      -> FROM        template   -> TEMPLATE     messages -> MESSAGE
#   system     -> SYSTEM      parameters -> PARAMETER    license  -> LICENSE
response = client.create(
  model='my-assistant',
  from_='gemma4',
  system='You are Mario from Super Mario Bros.',
  template='{{ .System }} {{ .Prompt }}',
  parameters={'temperature': 0.6, 'num_ctx': 4096, 'stop': ['<end_of_turn>']},
  messages=[
    {'role': 'user', 'content': 'Who are you?'},
    {'role': 'assistant', 'content': "It's-a me, Mario!"},
  ],
  license='MIT',
  stream=False,
)
print(response.status)

# To import model weights from a local GGUF file instead of deriving from an
# existing model, see create-gguf.py.
