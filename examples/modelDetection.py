"""List installed Ollama models and optionally remove selected models."""

import sys
from dataclasses import dataclass

from ollama import ResponseError
from ollama import delete as delete_model
from ollama import list as list_models


@dataclass(frozen=True)
class InstalledModel:
  name: str
  size: int
  parameter_size: str
  quantization: str


def format_size(size):
  """Format a byte count using binary units."""
  value = float(size)
  for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
    if value < 1024 or unit == 'TiB':
      return f'{value:.1f} {unit}'
    value /= 1024
  return f'{value:.1f} TiB'


def get_installed_models():
  """Return locally installed models reported by Ollama."""
  installed = []
  for model in list_models().models:
    if not model.model:
      continue

    details = model.details
    installed.append(
      InstalledModel(
        name=model.model,
        size=int(model.size) if model.size is not None else 0,
        parameter_size=details.parameter_size if details and details.parameter_size else '-',
        quantization=details.quantization_level if details and details.quantization_level else '-',
      )
    )
  return sorted(installed, key=lambda model: model.name.lower())


def print_models(models):
  """Print installed models and their estimated storage usage."""
  headers = ('#', 'Model', 'Size', 'Parameters', 'Quantization')
  rows = [(str(index), model.name, format_size(model.size), model.parameter_size, model.quantization) for index, model in enumerate(models, start=1)]
  widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]

  print('\nInstalled Ollama models:\n')
  print('  '.join(header.ljust(widths[index]) for index, header in enumerate(headers)))
  print('  '.join('-' * width for width in widths))
  for row in rows:
    print('  '.join(value.ljust(widths[index]) for index, value in enumerate(row)))

  total_size = sum(model.size for model in models)
  print(f'\nTotal reported size: {format_size(total_size)}')


def parse_selection(selection, models):
  """Convert a comma-separated numeric selection into unique models."""
  normalized = selection.strip().lower()
  if not normalized:
    return []
  if normalized == 'all':
    return list(models)

  selected = []
  seen = set()
  for value in normalized.split(','):
    value = value.strip()
    if not value.isdigit():
      raise ValueError(f'Invalid selection: {value!r}')

    index = int(value)
    if index < 1 or index > len(models):
      raise ValueError(f'Model number {index} is outside the available range')
    if index not in seen:
      selected.append(models[index - 1])
      seen.add(index)
  return selected


def ask_which_models(models):
  """Prompt until the user chooses valid model numbers or cancels."""
  while True:
    selection = input('\nEnter model numbers separated by commas, "all" to select everything,\nor press Enter to cancel: ')
    try:
      return parse_selection(selection, models)
    except ValueError as error:
      print(f'Error: {error}')


def confirm_removal(models):
  """Require explicit confirmation before deleting models."""
  print('\nSelected for removal:')
  for model in models:
    print(f'  - {model.name} ({format_size(model.size)})')
  selected_size = sum(model.size for model in models)
  print(f'\nSelected reported size: {format_size(selected_size)}')
  print('Actual freed space may be smaller because Ollama models can share data layers.')
  confirmation = input('Remove these models? Type "yes" to confirm: ')
  return confirmation.strip().lower() == 'yes'


def remove_models(models):
  """Remove each selected model and return the number of failures."""
  failures = 0
  for model in models:
    try:
      response = delete_model(model.name)
      status = response.status or 'success'
      print(f'Removed {model.name}: {status}')
    except (ConnectionError, ResponseError) as error:
      failures += 1
      print(f'Could not remove {model.name}: {error}', file=sys.stderr)
  return failures


def main():
  try:
    models = get_installed_models()
  except (ConnectionError, ResponseError, KeyboardInterrupt) as error:
    print(f'Could not connect to Ollama: {error}', file=sys.stderr)
    print('Make sure Ollama is installed and running.', file=sys.stderr)
    return 1

  if not models:
    print('No Ollama models are currently installed.')
    return 0

  print_models(models)
  try:
    selected = ask_which_models(models)
    if not selected:
      print('No models selected. Nothing was removed.')
      return 0
    if not confirm_removal(selected):
      print('Removal cancelled. Nothing was removed.')
      return 0
  except (EOFError, KeyboardInterrupt):
    print('\nRemoval cancelled. Nothing was removed.')
    return 0

  failures = remove_models(selected)
  if failures:
    print(f'Finished with {failures} removal failure(s).', file=sys.stderr)
    return 1

  print('\nSelected models were removed successfully.')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
