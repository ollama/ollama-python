from ollama import ShowResponse, show

response: ShowResponse = show('gemma3')
print('Model Information:')
print(f'Modified at:   {response.modified_at}')
print(f'Template:      {response.template}')
print(f'Modelfile:     {response.modelfile}')
print(f'License:       {response.license}')
print(f'Details:       {response.details}')
print(f'Model Info:    {response.modelinfo}')
print(f'Parameters:    {response.parameters}')
print(f'Capabilities:  {response.capabilities}')
