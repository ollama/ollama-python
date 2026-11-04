"""Compare runtime performance across locally installed Ollama models."""

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ollama import GenerateResponse, ResponseError, generate
from ollama import list as list_models

DEFAULT_PROMPT = 'Write a numbered list from 1 to 100, with each number followed by its English word.'
NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass
class BenchmarkResult:
  model: str
  run: int
  ttft_seconds: Optional[float]
  wall_seconds: float
  total_seconds: Optional[float]
  load_seconds: Optional[float]
  prompt_tokens: Optional[int]
  prompt_tokens_per_second: Optional[float]
  eval_tokens: Optional[int]
  eval_tokens_per_second: Optional[float]


def non_negative_int(value):
  parsed = int(value)
  if parsed < 0:
    raise argparse.ArgumentTypeError('must be zero or greater')
  return parsed


def positive_int(value):
  parsed = int(value)
  if parsed < 1:
    raise argparse.ArgumentTypeError('must be one or greater')
  return parsed


def nanoseconds_to_seconds(value):
  if value is None:
    return None
  return value / NANOSECONDS_PER_SECOND


def tokens_per_second(count, duration):
  if count is None or duration is None or duration <= 0:
    return None
  return count * NANOSECONDS_PER_SECOND / duration


def benchmark_once(model, prompt, num_predict, keep_alive, run):
  started = time.perf_counter()
  first_token_at: Optional[float] = None
  final_response: Optional[GenerateResponse] = None

  stream = generate(
    model=model,
    prompt=prompt,
    stream=True,
    keep_alive=keep_alive,
    options={
      'num_predict': num_predict,
      'seed': 42,
      'temperature': 0,
    },
  )

  for part in stream:
    if first_token_at is None and (part.response or part.thinking):
      first_token_at = time.perf_counter()
    final_response = part

  finished = time.perf_counter()
  if final_response is None:
    raise RuntimeError('Ollama returned an empty response stream')

  return BenchmarkResult(
    model=model,
    run=run,
    ttft_seconds=None if first_token_at is None else first_token_at - started,
    wall_seconds=finished - started,
    total_seconds=nanoseconds_to_seconds(final_response.total_duration),
    load_seconds=nanoseconds_to_seconds(final_response.load_duration),
    prompt_tokens=final_response.prompt_eval_count,
    prompt_tokens_per_second=tokens_per_second(final_response.prompt_eval_count, final_response.prompt_eval_duration),
    eval_tokens=final_response.eval_count,
    eval_tokens_per_second=tokens_per_second(final_response.eval_count, final_response.eval_duration),
  )


def median(values):
  present = [value for value in values if value is not None]
  return statistics.median(present) if present else None


def summarize(model, results):
  return {
    'model': model,
    'runs': len(results),
    'median_ttft_seconds': median(result.ttft_seconds for result in results),
    'median_wall_seconds': median(result.wall_seconds for result in results),
    'median_total_seconds': median(result.total_seconds for result in results),
    'median_load_seconds': median(result.load_seconds for result in results),
    'median_prompt_tokens_per_second': median(result.prompt_tokens_per_second for result in results),
    'median_eval_tokens_per_second': median(result.eval_tokens_per_second for result in results),
    'median_eval_tokens': median(None if result.eval_tokens is None else float(result.eval_tokens) for result in results),
  }


def format_seconds(value):
  if not isinstance(value, (float, int)):
    return '-'
  return f'{value:.3f}s'


def format_rate(value):
  if not isinstance(value, (float, int)):
    return '-'
  return f'{value:.1f}'


def format_count(value):
  if not isinstance(value, (float, int)):
    return '-'
  return f'{value:.0f}'


def print_table(summaries):
  headers = ('Model', 'Runs', 'TTFT', 'Total', 'Load', 'Prompt tok/s', 'Gen tok/s', 'Gen tokens')
  rows: List[Tuple[str, ...]] = []
  for summary in summaries:
    rows.append(
      (
        str(summary['model']),
        str(summary['runs']),
        format_seconds(summary['median_ttft_seconds']),
        format_seconds(summary['median_total_seconds']),
        format_seconds(summary['median_load_seconds']),
        format_rate(summary['median_prompt_tokens_per_second']),
        format_rate(summary['median_eval_tokens_per_second']),
        format_count(summary['median_eval_tokens']),
      )
    )

  widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
  print('\n' + '  '.join(header.ljust(widths[index]) for index, header in enumerate(headers)))
  print('  '.join('-' * width for width in widths))
  for row in rows:
    print('  '.join(value.ljust(widths[index]) for index, value in enumerate(row)))


def write_csv(path, summaries):
  fieldnames = list(summaries[0])
  with path.open('w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summaries)


def parse_args():
  parser = argparse.ArgumentParser(
    description='Compare runtime performance across locally installed Ollama models.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument('models', nargs='*', help='installed model names; omit to benchmark every installed model')
  parser.add_argument('--prompt', default=DEFAULT_PROMPT, help='prompt sent to every model')
  parser.add_argument('--runs', type=positive_int, default=3, help='measured runs per model')
  parser.add_argument('--warmup', type=non_negative_int, default=1, help='unmeasured warm-up runs per model')
  parser.add_argument('--num-predict', type=positive_int, default=128, help='maximum generated tokens per run')
  parser.add_argument('--keep-alive', default='5m', help='how long Ollama keeps each model loaded')
  parser.add_argument('--csv', type=Path, help='write median summary metrics to this CSV path')
  return parser.parse_args()


def select_models(requested):
  installed = sorted(model.model for model in list_models().models if model.model)
  if not installed:
    raise RuntimeError('No local models are installed. Run `ollama pull <model>` first.')

  if not requested:
    return installed

  missing = sorted(set(requested) - set(installed))
  if missing:
    raise RuntimeError(f'Models are not installed: {", ".join(missing)}')
  return list(dict.fromkeys(requested))


def main():
  args = parse_args()
  try:
    models = select_models(args.models)
  except (ConnectionError, ResponseError, RuntimeError) as error:
    print(f'Error: {error}', file=sys.stderr)
    return 1

  print(f'Benchmarking {len(models)} model(s) with {args.runs} measured run(s) each.')
  print(f'Prompt: {args.prompt!r}')
  print(f'Max generated tokens: {args.num_predict}; warm-up runs: {args.warmup}')
  print('Keep hardware load and background activity stable for meaningful comparisons.')

  results_by_model: Dict[str, List[BenchmarkResult]] = {}
  for model in models:
    print(f'\n{model}')
    try:
      for warmup in range(1, args.warmup + 1):
        print(f'  warm-up {warmup}/{args.warmup}...', end='', flush=True)
        benchmark_once(model, args.prompt, args.num_predict, args.keep_alive, run=0)
        print(' done')

      model_results = []
      for run in range(1, args.runs + 1):
        print(f'  run {run}/{args.runs}...', end='', flush=True)
        result = benchmark_once(model, args.prompt, args.num_predict, args.keep_alive, run=run)
        model_results.append(result)
        print(f' {format_rate(result.eval_tokens_per_second)} tokens/s')
      results_by_model[model] = model_results
    except (ConnectionError, ResponseError, RuntimeError) as error:
      print(f' failed: {error}', file=sys.stderr)

  summaries = [summarize(model, results) for model, results in results_by_model.items()]
  if not summaries:
    print('No model completed the benchmark.', file=sys.stderr)
    return 1

  summaries.sort(key=lambda summary: float(summary['median_eval_tokens_per_second'] or 0), reverse=True)
  print_table(summaries)

  if args.csv:
    write_csv(args.csv, summaries)
    print(f'\nWrote summary CSV to {args.csv}')

  return 0


if __name__ == '__main__':
  raise SystemExit(main())
