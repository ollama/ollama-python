#!/usr/bin/env python3

import sys
import requests
import json


def call_ollama(prompt, model="tinyllama", base_url="http://localhost:11434", stream=True):
    """
    Sends a prompt to the Ollama API and streams the response back.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }

    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    yield data.get("response", "")
        else:
            yield response.json().get("response", "No response received")

    except requests.RequestException as e:
        yield f"\n[Error] Could not connect to Ollama. Is it running?\nDetails: {str(e)}\n"


def main():
    """
    Main entry point for CLI usage.
    """
    # Get prompt from command-line or input
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
    else:
        prompt = input("Enter your prompt: ")

    print("Thinking...\nOllama: ", end="", flush=True)

    # Call Ollama and print the output in real-time
    for chunk in call_ollama(prompt):
        print(chunk, end="", flush=True)

    print()  # Final newline


if __name__ == "__main__":
    main()

