from ollama import ListResponse, list

response: ListResponse = list()

for model in response.models:
  print('Name:', model.model)
  print('  Size (MB):', f'{(model.size.real / 1024 / 1024):.2f}')
  if model.details:
    print('  Format:', model.details.format)
    print('  Family:', model.details.family)
    print('  Parameter Size:', model.details.parameter_size)
    print('  Quantization Level:', model.details.quantization_level)
  print('\n')
