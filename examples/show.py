from ollama import ShowResponse, show
from ollama import ListResponse, list

response: ListResponse = list()

for model in response.models:
  print('Name:', model.model)
  model_details: ShowResponse = show(model.model)
  if "embedding" in model_details.capabilities:
    print("Embedding model")
  elif "vision" in model_details.capabilities:
    print("Vision model")
  elif "completion" in model_details.capabilities:
    print("Chat model")
  else:
    print("Unknown", model_details.capabilities)
  print('\n')
