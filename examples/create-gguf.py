# Import a local GGUF file as a model.
#
# Modelfile equivalent:
#   FROM ./model.gguf
#
# The API takes blob digests rather than file paths, so each file is uploaded
# first with Client.create_blob(), which streams the file to the server and
# returns its 'sha256:...' digest. The same pattern works for adapters=.

from ollama import Client

client = Client()

path = 'path/to/model.gguf'  # replace with a real GGUF file on disk

response = client.create(
  model='my-gguf-model',
  files={'model.gguf': client.create_blob(path)},
  # quantize='q4_K_M',  # optional: quantize the weights during import
  system='You are a helpful assistant.',
  stream=False,
)
print(response.status)
