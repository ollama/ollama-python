"""Multimodal chat with image input.

Sends an image alongside a text prompt, allowing the model to describe
or answer questions about the image content.

Prerequisites:
    ollama pull gemma3

Usage:
    python multimodal-chat.py
"""

from ollama import chat

# from pathlib import Path

# Pass in the path to the image
path = input('Please enter the path to the image: ')

# You can also pass in base64 encoded image data
# img = base64.b64encode(Path(path).read_bytes()).decode()
# or the raw bytes
# img = Path(path).read_bytes()

response = chat(
  model='gemma3',
  messages=[
    {
      'role': 'user',
      'content': 'What is in this image? Be concise.',
      'images': [path],
    }
  ],
)

print(response.message.content)
