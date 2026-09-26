import sys
sys.path.insert(0, '.')  # force Python to use local folder first

from ollama._types import ShowResponse

test_data = {
    "template": "test template",
    "details": None,
}

response = ShowResponse(**test_data)
print(f"modelinfo: {response.modelinfo}")
print("✓ Fix works — no ValidationError")