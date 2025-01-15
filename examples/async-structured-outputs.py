import asyncio

from pydantic import BaseModel

from ollama import AsyncClient


# Define the schema for the response
class FriendInfo(BaseModel):
  name: str
  age: int
  is_available: bool


class FriendList(BaseModel):
  friends: list[FriendInfo]


async def main():
  client = AsyncClient()
  response = await client.chat(
    model='llama3.1:8b',
    messages=[{'role': 'user', 'content': 'I have two friends. The first is Ollama 22 years old busy saving the world, and the second is Alonso 23 years old and wants to hang out. Return a list of friends in JSON format'}],
    format=FriendList.model_json_schema(),  # Use Pydantic to generate the schema
    options={'temperature': 0},  # Make responses more deterministic
  )

  # Use Pydantic to validate the response
  friends_response = FriendList.model_validate_json(response.message.content)
  print(friends_response)


if __name__ == '__main__':
  asyncio.run(main())
