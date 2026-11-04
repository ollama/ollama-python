from ollama import Client


def batch_test_ollama_models():
    client = Client()
    
    try:
        available_models_info = client.list()
        available_models = [m.model if hasattr(m, 'model') else m.get('model', m.get('name', '')) for m in available_models_info.get('models', [])]
    except Exception as e:
        print(f"Error listing models: {e}")
        available_models = []
    
    # Custom list of desired models to test - filtered by availability
    models_to_compare = [
        "gemini-3-flash-preview:latest",
        "llama3.1:latest",
        "gemma:latest",
        "qwen3:latest",
        "devstral-2:123b-cloud",
    ]
    
    # Verify models are actually installed/registered
    final_model_list = [m for m in models_to_compare if m in available_models]

    # Or use all available models excluding a list of models to exclude
    # models = [m for m in available_models if m not in ["model1", "model2", "model3"]]

    if not final_model_list and available_models:
        print("None of the target models are installed. Using all available model for testing.")
        final_model_list = available_models
    elif not available_models:
        print("No models found in Ollama. Please pull some models first.")
        exit(1)
    
    synthesis_model = final_model_list[0] # Set to a long-model else default to first in list
    
    responses = []  # List of strings for each model
    synthesized_history = [] # History of final responses
    
    while True:
        prompt = input("\nEnter prompt or X to quit: ")
        if prompt.upper() == "X":
            break
           
        # Reset responses for new prompt cycle
        responses = [[] for _ in range(len(final_model_list))]

        for i, model in enumerate(final_model_list):
            print(f"\n--- Model [{i+1}/{len(final_model_list)}]: {model} ---")
            in_thinking = False
            try:
                for part in client.chat(model=model, messages=[{"role": "user", "content": prompt}], stream=True):
                    
                    # Handle thinking block if present
                    if hasattr(part.message, 'thinking') and part.message.thinking:
                        if not in_thinking:
                            print("\nThinking: ", end="", flush=True)
                            in_thinking = True
                        print(part.message.thinking, end="", flush=True)
                        # [i].append(part.message.thinking) # Don't append thinking to responses

                    # Handle actual content
                    if part.message.content:
                        if in_thinking:
                            print("\nContent: ", end="", flush=True)
                            in_thinking = False
                        print(part.message.content, end="", flush=True)
                        responses[i].append(part.message.content) # Collect responses to save to file or do synthesis/summary/evaluation on all of them
            except Exception as e:
                print(f"\nError with model {model}: {e}")
        
        print("\n\n" + "="*50)
        print("SYNTHESIZING RESPONSES using", synthesis_model)
        print("="*50 + "\n")

        # Prepare data for synthesis
        full_responses_text = ""

        for idx, model_name in enumerate(final_model_list):
            resp_text = "".join(responses[idx])
            full_responses_text += f"""
            Model: {model_name}
            Response: {resp_text}
            """

        # PROMPT FOR PROCESSING ALL RESPONSES TOGETHER - Edit as you see fit
        synthesis_prompt = f"Here are several outputs for the following input: '{prompt}':\n\n{full_responses_text}\nSynthesize these outputs into one distilled but comprehensive, high-quality response that consolidates all of the unique points in their best form from all of the outputs, avoiding redundancy and optimizing for quality, clarity and effectiveness: "

        current_synthesis = []
        try:
            for part in client.chat(model=synthesis_model, messages=[{"role": "user", "content": synthesis_prompt}], stream=True):
                if hasattr(part.message, 'thinking') and part.message.thinking:
                    print(part.message.thinking, end="", flush=True)
                if part.message.content:
                    print(part.message.content, end="", flush=True)
                    current_synthesis.append(part.message.content)
            synthesized_history.append("".join(current_synthesis))
        except Exception as e:
            print(f"\nError during synthesis: {e}")
        print("\n" + "-"*50 + "\n")

def main():
    batch_test_ollama_models()

if __name__ == "__main__":
    main()