from ollama import Client

client = Client()
response = client.create(model='my-assistant', from_='llama3.2', stream=False)
print(response.status)
