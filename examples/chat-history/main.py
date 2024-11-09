import ollama

messages = []

def send(chat):
  messages.append(
    {
      'role': 'user',
      'content': chat,
    }
  )
  stream = ollama.chat(
    model = 'llama3.1:latest',
    messages = messages,
    stream = True,
  )

  response = ""
  for chunk in stream:
    part = chunk['message']['content']
    print(part, end='', flush=True)
    response = response + part

  messages.append(
    {
      'role': 'assistant',
      'content': response,
    }
  )

  print("")

print("type '/exit' to quit, '/history' to dump message history.")

while True:
    chat = input(">>> ")

    if chat == "/exit":
        break
    elif chat == "/history":
        print(messages)
    elif len(chat) > 0:
        send(chat)