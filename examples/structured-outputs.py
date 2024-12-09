from ollama import chat
from pydantic import BaseModel

# Define the schema for the response
class FriendInfo(BaseModel):
    name: str
    age: int
    is_available: bool

class FriendList(BaseModel):
    friends: list[FriendInfo]

# Send a request to the chat model
response = chat(
    model='llama3.2',
    messages=[{
        'role': 'user',
        'content': (
            'I have two friends. The first is Ollama, 22 years old, busy saving the world, '
            'and the second is Alonso, 23 years old, and wants to hang out. '
            'Return a list of friends in JSON format.'
        )
    }],
    format=FriendList.model_json_schema(),  # Use Pydantic to generate the schema
    options={'temperature': 0},  # Make responses more deterministic
)

# Print the raw response to understand its structure
print("Raw Response:", response)

# Use Pydantic to validate and parse the response
# Ensure the key access (["message"]["content"]) matches the actual structure of the API response.
if "message" in response and "content" in response["message"]:
    friends_response = FriendList.model_validate_json(response["message"]["content"])
    print(friends_response)
else:
    print("Unexpected response structure:", response)
