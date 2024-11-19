import asyncio
import ollama


async def main():
  client = ollama.AsyncClient()

  response = await client.list()
  models = response['models']
  models_data = []
  for model in models:
    if model.get('details'):  # Check if details exist
      models_data.append(
        (
          model.get('name', 'N/A'),
          model.get('size', 0) / 1024 / 1024,  # Convert to MB
          model.get('details', {}).get('format', 'N/A'),
          model.get('details', {}).get('family', 'N/A'),
          model.get('details', {}).get('parameter_size', 'N/A'),
          model.get('details', {}).get('quantization_level', 'N/A'),
        )
      )

  print(f'\n{len(models)} models found!')
  print('\nDetailed model information:')
  for model in models_data:
    print(f'Name: {model[0]}')
    print(f'Size (MB): {model[1]:.2f}')
    print(f'Format: {model[2]}')
    print(f'Family: {model[3]}')
    print(f'Parameter Size: {model[4]}')
    print(f'Quantization Level: {model[5]}')
    print('-' * 50)


if __name__ == '__main__':
  asyncio.run(main())
