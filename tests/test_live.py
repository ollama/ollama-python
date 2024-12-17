"""Tests of making requests to a live Ollama server running locally"""

import os

import pytest
from pydantic import BaseModel

from ollama import AsyncClient

pytestmark = [
  pytest.mark.skipif(not os.getenv('OLLAMA_TEST_LIVE'), reason='live tests disabled'),
  pytest.mark.asyncio,
]


async def test_text():
  client = AsyncClient()
  response = await client.chat(
    'qwen2:0.5b',
    messages=[{'role': 'user', 'content': 'What is the capital of France?'}],
    options={'temperature': 0},
  )
  print('Text response:', response)
  assert 'paris' in response.message.content.lower()


async def test_stream():
  client = AsyncClient()
  stream = await client.chat(
    'qwen2:0.5b',
    messages=[{'role': 'user', 'content': 'What is the capital of France?'}],
    options={'temperature': 0},
    stream=True,
  )
  text = ''.join([part['message']['content'] async for part in stream])
  print('Text response:', text)
  assert 'paris' in text.lower()


class MyModel(BaseModel):
  city: str


async def test_structured():
  client = AsyncClient()
  response = await client.chat(
    'qwen2:0.5b',
    messages=[{'role': 'user', 'content': 'What is the capital of France?'}],
    format=MyModel.model_json_schema(),
    options={'temperature': 0},
  )
  my_model = MyModel.model_validate_json(response.message.content)
  assert my_model.city.lower() == 'paris'
