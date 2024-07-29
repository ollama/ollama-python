import ollama


def get_answer(model, prompt):
    print(f'Creating the result from {model} for you. Please wait ...')

    ollama.pull(model)

    response = ollama.chat(model, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    result = response['message']['content']

    information = f'\r\nThe answer from{model}:\r\n{result}'

    print(information)

    with open('output.txt', 'a') as f:
        f.write(information)

    print(f'{model} done!')


question = '''
    Why the sky is blue? 
'''.strip()

models = ['gemma2', 'internlm2', 'llama3']

[get_answer(model, question) for model in models]
