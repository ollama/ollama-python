from ollama import Client

client = Client()
response = client.create(
  model='my-assistant',
  from_='gemma3',
  system='You are mario from Super Mario Bros.',
  stream=False,
)
print(response.status)
