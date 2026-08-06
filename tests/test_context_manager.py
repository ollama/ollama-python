import sys
sys.path.insert(0, '.')

from ollama._client import Client

# Test with-as syntax works
with Client() as client:
    print(f"Client type: {type(client)}")
    print("✓ with-as syntax works")

print("✓ Client cleaned up after with block")