from ollama import Client

client = Client()
path = ''
# Passing in wrong path for image error sucks
response = client.chat(
  model='llama3.2-vision',
  messages=[
    {
      'role': 'user',
      'content': 'What is in this image? Be concise. Respond with the structure {"focal": "...", "subject": "...", "background": "..."}',
      'images': [path],
    }
  ],
)

print(response.message.content)
