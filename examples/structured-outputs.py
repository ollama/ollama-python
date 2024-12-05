from dataclasses import dataclass
import json
from ollama import chat
from pydantic import BaseModel


# Using Pydantic
## Define the schema for the response
class FriendInfo(BaseModel):
  name: str
  age: int
  is_available: bool


class FriendList(BaseModel):
  friends: list[FriendInfo]


response = chat(
  model='llama3.1:8b',
  messages=[{'role': 'user', 'content': 'I have two friends. The first is Ollama 22 years old busy saving the world, and the second is Alonso 23 years old and wants to hang out. Return a list of friends in JSON format'}],
  format=FriendList.model_json_schema(),  # Use Pydantic to generate the schema
  options={'temperature': 0},  # Make responses more deterministic
)

## Use Pydantic to validate the response
friends_response = FriendList.model_validate_json(response.message.content)
print(friends_response)


# Using Dataclass
## Define the schema for the response
@dataclass
class FriendInfo:
  name: str
  age: int
  is_available: bool


@dataclass
class FriendList:
  friends: list[FriendInfo]


friends_list_schema = {'type': 'object', 'properties': {'friends': {'type': 'array', 'items': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}, 'is_available': {'type': 'boolean'}}, 'required': ['name', 'age', 'is_available']}}}, 'required': ['friends']}

## Serialize the dataclass schema using asdict
response = chat(
  model='llama3.1:8b',
  messages=[{'role': 'user', 'content': 'I have two friends. The first is Ollama 22 years old busy saving the world, and the second is Alonso 23 years old and wants to hang out. Return a list of friends in JSON format'}],
  format=friends_list_schema,
  options={'temperature': 0},
)

friends_response = FriendList(friends=[FriendInfo(**friend) for friend in json.loads(response.message.content)['friends']])
print(friends_response)
