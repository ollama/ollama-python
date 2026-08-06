import sys
sys.path.insert(0, '.')

from ollama._types import Message

# Test with id
tc1 = Message.ToolCall(
    id="call_abc123",
    function=Message.ToolCall.Function(
        name="get_weather",
        arguments={"city": "Hyderabad"}
    )
)
print(f"id: {tc1.id}")
print(f"function: {tc1.function.name}")

# Test without id (backward compatibility)
tc2 = Message.ToolCall(
    function=Message.ToolCall.Function(
        name="get_weather",
        arguments={"city": "Hyderabad"}
    )
)
print(f"id default: {tc2.id}")
print("✓ Both work")