import json

from ollama import list

model_list = list()

# print nicely with json format
formatted_json = json.dumps(model_list, indent=4)
print(formatted_json)

print('\n')

# print only available models
name_list = [d.get('name') for d in model_list['models']]
print(name_list)

print('\n')

# print key-value pair
for contents in model_list['models']:
    for key, value in contents.items():
        print(key, value)
