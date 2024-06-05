from ollama import ps, pull, chat

pull('mistral')
chat('mistral', messages=[{'role': 'user', 'content': 'Pick a number'}])

response = ps()

name = response['models'][0]['name']
size = response['models'][0]['size']
size_vram = response['models'][0]['size_vram']

if size == size_vram:
    print(f'{name}: 100% GPU')
elif not size_vram:
    print(f'{name}: 100% CPU')
else:
    size_cpu = size - size_vram
    cpu_percent = round(size_cpu / size * 100)
    print(f'{name}: {cpu_percent}% CPU/{100 - cpu_percent}% GPU')
