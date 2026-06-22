from ollama import Client

client = Client()

modelfile = '''
FROM gemma3
SYSTEM You are mario from Super Mario Bros.
'''

response = client.create(
  model='my-assistant',
  modelfile=modelfile,
  stream=False,
)
print(response.status)
