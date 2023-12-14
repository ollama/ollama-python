from ollama.client import Client

_default_client = Client()

def generate(*args, **kwargs):
  return _default_client.generate(*args, **kwargs)

def chat(*args, **kwargs):
  return _default_client.chat(*args, **kwargs)

def pull(*args, **kwargs):
  return _default_client.pull(*args, **kwargs)

def push(*args, **kwargs):
  return _default_client.push(*args, **kwargs)

def create(*args, **kwargs):
  return _default_client.create(*args, **kwargs)

def delete(*args, **kwargs):
  return _default_client.delete(*args, **kwargs)

def list(*args, **kwargs):
  return _default_client.list(*args, **kwargs)

def copy(*args, **kwargs):
  return _default_client.copy(*args, **kwargs)

def show(*args, **kwargs):
  return _default_client.show(*args, **kwargs)
