import ollama

def persian_chatbot():
    """
    یک چت‌بات فارسی تعاملی با قابلیت استریم و حافظه مکالمه
    برای تست: python examples/persian_chatbot.py
    """
    print("چت‌بات فارسی Ollama (برای خروج 'خروج' بنویسید)")
    print("="*50)
    
    messages = [
        {
            'role': 'system', 
            'content': 'تو یک دستیار فارسی باهوش، مودب و دقیق هستی. همیشه به زبان فارسی جواب بده.'
        }
    ]
    
    while True:
        user_input = input("\nشما: ")
        if user_input.lower() in ['خروج', 'exit', 'q', 'quit']:
            print("خدانگهدار!")
            break
        
        messages.append({'role': 'user', 'content': user_input})
        
        print("ربات: ", end="")
        full_response = ""
        
        try:
            stream = ollama.chat(model='llama3.1:8b', messages=messages, stream=True)
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                full_response += content
            print()
        except Exception as e:
            print(f"\nخطا: {e}")
            print("آیا مطمئنید ollama serve روشن است؟")
            continue
        
        messages.append({'role': 'assistant', 'content': full_response})

if __name__ == "__main__":
    persian_chatbot()