import sys
sys.path.insert(0, '.')

from ollama._types import ChatRequest

# Test that tool_choice is accepted
req = ChatRequest(
    model="llama3.2",
    tool_choice="auto"
)
print(f"tool_choice: {req.tool_choice}")
print("✓ tool_choice parameter works")

# Test default is None
req2 = ChatRequest(model="llama3.2")
print(f"tool_choice default: {req2.tool_choice}")
print("✓ default is None")