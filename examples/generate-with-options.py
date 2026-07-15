"""
Example demonstrating how to use options with ollama.generate() to control model behavior.

This example shows how different options affect:
- Creativity and randomness (temperature, top_p)
- Response length (num_predict)
- Reproducibility (seed)
- Stop conditions (stop sequences)
- Repetition control (repeat_penalty)
"""

import ollama
from ollama import Options


def main():
    model = 'llama3.2'
    base_prompt = 'Write a creative opening line for a sci-fi novel about space exploration'
    
    print("=== Ollama Generate Options Examples ===\n")
    
    # Example 1: Creative, longer response
    print("1. Creative Response (high temperature, diverse sampling)")
    print("-" * 55)
    
    creative_options = Options(
        temperature=1.0,        # High creativity
        top_p=0.9,             # Allow diverse word choices
        num_predict=80,        # Longer response
        repeat_penalty=1.1,    # Avoid repetition
        top_k=40              # Consider top 40 tokens
    )
    
    try:
        response = ollama.generate(
            model=model,
            prompt=base_prompt,
            options=creative_options
        )
        print(f"Response: {response['response']}")
        print(f"Tokens generated: {response.get('eval_count', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Focused, deterministic response
    print("2. Focused Response (low temperature, reproducible)")
    print("-" * 55)
    
    focused_options = {
        'temperature': 0.3,     # Low creativity, more focused
        'top_p': 0.5,          # More conservative word choices
        'num_predict': 40,     # Shorter response
        'seed': 42,            # Reproducible results
        'repeat_penalty': 1.05  # Light repetition penalty
    }
    
    try:
        response = ollama.generate(
            model=model,
            prompt=base_prompt,
            options=focused_options
        )
        print(f"Response: {response['response']}")
        print(f"Tokens generated: {response.get('eval_count', 'N/A')}")
        print("Note: Run this again with the same seed for identical results!")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Controlled response with stop sequences
    print("3. Controlled Response (with stop sequences)")
    print("-" * 55)
    
    list_prompt = "List the top 5 benefits of renewable energy sources:"
    
    controlled_options = Options(
        temperature=0.7,
        num_predict=200,
        stop=['\n6.', 'In conclusion', 'However', 'Additionally'],  # Stop at these phrases
        repeat_penalty=1.1
    )
    
    try:
        response = ollama.generate(
            model=model,
            prompt=list_prompt,
            options=controlled_options
        )
        print(f"Prompt: {list_prompt}")
        print(f"Response: {response['response']}")
        print("Note: Generation stopped early due to stop sequences")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 4: Comparison of different temperatures
    print("4. Temperature Comparison")
    print("-" * 55)
    
    temperatures = [0.1, 0.7, 1.2]
    comparison_prompt = "Describe the color blue in one sentence"
    
    for temp in temperatures:
        print(f"\nTemperature {temp}:")
        
        try:
            response = ollama.generate(
                model=model,
                prompt=comparison_prompt,
                options={
                    'temperature': temp,
                    'num_predict': 30,
                    'seed': 123  # Same seed for fair comparison
                }
            )
            print(f"  {response['response'].strip()}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 5: Context window demonstration (simplified)
    print("5. Context Window Control (simplified)")
    print("-" * 55)
    
    prompt = "Given this background: Solar power uses sunlight, wind power uses wind, hydroelectric uses water, geothermal uses Earth's heat. Which is best for a desert?"
    options = {
        'temperature': 0.5,
        'num_ctx': 512,
        'num_predict': 50
    }
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options=options
        )
        print(f"Prompt: {prompt}")
        print(f"Response: {response['response']}")
    except Exception as e:
        print(f"Error: {e}")
    print("\n" + "="*60)
    print("Examples completed! Try modifying the options to see different behaviors.")


if __name__ == '__main__':
    main()
