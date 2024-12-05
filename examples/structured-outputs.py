from ollama import chat
from pydantic import BaseModel
from rich import print


# Define the schema for the response
class FriendInfo(BaseModel):
  name: str
  age: int
  is_available: bool


class FriendList(BaseModel):
  friends: list[FriendInfo]


# Can also use manually defined schema directly
# schema = {'type': 'object', 'properties': {'friends': {'type': 'array', 'items': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}, 'is_available': {'type': 'boolean'}}, 'required': ['name', 'age', 'is_available']}}}, 'required': ['friends']}

response = chat(
  model='llama3.1:8b',
  messages=[{'role': 'user', 'content': 'I have two friends. The first is Ollama 22 years old busy saving the world, and the second is Alonso 23 years old and wants to hang out. Return a list of friends in JSON format'}],
  format=FriendList,  # or format=schema
  options={'temperature': 0},  # Make responses more deterministic
)

# Use Pydantic to validate the response
friends_response = FriendList.model_validate_json(response.message.content)
print(friends_response)
