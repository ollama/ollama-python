from ollama import chat, Message, Role
messages = [
    Message(role=Role.USER, content='Why is the sky blue?'),
]

response = chat('llama2', messages=messages)
print(response['message']['content'])
