from ollama import ShowResponse, show

response: ShowResponse = show("gemma3")
print(f"Modified at: {response.modified_at}\n")
print(f"Template: {response.template}\n")
print(f"Modelfile: {response.modelfile}\n")
print(f"License: {response.license}\n")
print(f"Details: {response.details}\n")
print(f"Model Info: {response.modelinfo}\n")
print(f"Parameters: {response.parameters}\n")
print(f"Capabilities: {response.capabilities}\n")