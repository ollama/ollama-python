import asyncio
import ollama


async def main():
  client = ollama.AsyncClient()

  response = await client.list()
  for model in response.models:
    if model.details:
      print(f'Name: {model.model}')
      print(f'Size (MB): {(model.size.real / 1024 / 1024):.2f}')
      print(f'Format: {model.details.format}')
      print(f'Family: {model.details.family}')
      print(f'Parameter Size: {model.details.parameter_size}')
      print(f'Quantization Level: {model.details.quantization_level}')
      print('-' * 50)


if __name__ == '__main__':
  asyncio.run(main())
