from ollama._client import Client, AsyncClient, Message, Options

__all__ = [
  'Client',
  'AsyncClient',
  'Message',
  'Options',
  'generate',
  'chat',
  'pull',
  'push',
  'create',
  'delete',
  'list',
  'copy',
  'show',
]

_default_client = Client()

generate = _default_client.generate
chat = _default_client.chat
pull = _default_client.pull
push = _default_client.push
create = _default_client.create
delete = _default_client.delete
list = _default_client.list
copy = _default_client.copy
show = _default_client.show

_async_default_client = AsyncClient()

async_generate = _async_default_client.generate
async_chat = _async_default_client.chat
async_pull = _async_default_client.pull
async_push = _async_default_client.push
async_create = _async_default_client.create
async_delete = _async_default_client.delete
async_list = _async_default_client.list
async_copy = _async_default_client.copy
async_show = _async_default_client.show
