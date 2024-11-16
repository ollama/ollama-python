"""
Example from ollama-python GitHub repo, extended for a vision model with system prompt, temperature setting added.
"""

import ollama

llm_output = ollama.generate(
    model="llama3.2-vision",
    system="Answer questions with brevity and conciseness.",
    prompt="Describe the image",
    images=["path/to/image.jpg"],
    options={"temperature": 0.1},
)

print(llm_output["response"])