from ollama import generate

prefix = '''def remove_non_ascii(s: str) -> str:
    """ '''

suffix = """
    return result
"""


response = generate(
  model='codellama:7b-code',
  prompt=f'<PRE> {prefix} <SUF>{suffix} <MID>',
  options={
    'num_predict': 128,
    'temperature': 0,
    'top_p': 0.9,
    'stop': ['<EOT>'],
  },
)

print(response['response'])
