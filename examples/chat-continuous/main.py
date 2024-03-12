import ollama

def get_response(message_history):
    model_name = 'qwen:7b'
    try:
        response = ollama.chat(model=model_name, messages=message_history, stream=False)
        received_message = response['message']
        return received_message['content'], received_message
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}

message_history = []
while True:
    user_input = input("Input (or type 'exit' to quit):")
    if user_input.lower() == 'exit':
        break
    message = {'role': 'user', 'content': user_input}
    message_history.append(message)
    output_text, received_message = get_response(message_history)
    if output_text:
        message_history.append(received_message)
        print(output_text)
    else:
        print("Failed to get response.")
