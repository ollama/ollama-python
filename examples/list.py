from ollama import list
from ollama._types import ListResponse

response: ListResponse = list()

for model in response.models:
  if model.details:
    print(f'Name: {model.model}')
    print(f'Size (MB): {(model.size.real / 1024 / 1024):.2f}')
    print(f'Format: {model.details.format}')
    print(f'Family: {model.details.family}')
    print(f'Parameter Size: {model.details.parameter_size}')
    print(f'Quantization Level: {model.details.quantization_level}')
    print('-' * 50)
