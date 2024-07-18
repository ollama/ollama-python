from ollama import generate

response = generate(
  model='codellama:7b-code',
  prompt='def remove_non_ascii(s: str) -> str:',
  suffix='return result',

  options={
    'num_predict': 128,
    'temperature': 0,
    'top_p': 0.9,
    'stop': ['<EOT>'],
  },
)

print(response['response'])
