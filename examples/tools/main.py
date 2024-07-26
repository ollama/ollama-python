import json
import ollama
import asyncio
from typing import Literal, TypedDict

registry = ollama.ToolRegistry()

# Simulates an API call to get flight times
# In a real application, this would fetch data from a live database or API
@registry.register
def get_flight_times(departure: str, arrival: str) -> str:
  """
  Get flight times.
  :param departure: Departure location code
  :param arrival: Arrival location code
  """
  flights = {
    'NYC-LAX': {'departure': '08:00 AM', 'arrival': '11:30 AM', 'duration': '5h 30m'},
    'LAX-NYC': {'departure': '02:00 PM', 'arrival': '10:30 PM', 'duration': '5h 30m'},
    'LHR-JFK': {'departure': '10:00 AM', 'arrival': '01:00 PM', 'duration': '8h 00m'},
    'JFK-LHR': {'departure': '09:00 PM', 'arrival': '09:00 AM', 'duration': '7h 00m'},
    'CDG-DXB': {'departure': '11:00 AM', 'arrival': '08:00 PM', 'duration': '6h 00m'},
    'DXB-CDG': {'departure': '03:00 AM', 'arrival': '07:30 AM', 'duration': '7h 30m'},
  }

  key = f'{departure}-{arrival}'.upper()
  return json.dumps(flights.get(key, {'error': 'Flight not found'}))

@registry.register
class User(TypedDict): # This can also be `pydantic.BaseModel`, or `typing.NamedTuple`.
    """
    User Information
    :param name: Name of the user
    :param role: Role assigned to the user 
    """
    name: str
    role: Literal['admin', 'developer', 'tester']

# Or you can just call `registry.register_multiple(get_flight_times, User)`

# Get tools schema
tools = registry.tools
print(json.dumps(tools, indent=3))

async def run(model: str, query: str):
  client = ollama.AsyncClient()
  # Initialize conversation with a user query
  messages = [{'role': 'user', 'content': query}]

  # First API call: Send the query and function description to the model
  response = await client.chat(
    model=model, messages=messages, tools=tools,
  )

  # Add the model's response to the conversation history
  messages.append(response['message'])

  # Check if the model decided to use the provided function
  tool_calls = response['message'].get('tool_calls')
  if not tool_calls:
    print("The model didn't use the function. Its response was:")
    print(response['message']['content'])
    return

  # Process function calls made by the model
  for tool in response['message']['tool_calls']:
    function_response = registry.invoke(**tool['function'])
    print(f"{function_response=}")
    
    messages.append(
      {
        'role': 'tool',
        'content': str(function_response),
      }
    )

  # Second API call: Get final response from the model
  final_response = await client.chat(model=model, messages=messages)
  print(final_response['message']['content'])


# Run the async function
model = 'llama3-groq-tool-use:latest'
asyncio.run(run(
  model=model,
  query='What is the flight time from New York (NYC) to Los Angeles (LAX)?'
))
"""It should output something like:
function_response='{"departure": "08:00 AM", "arrival": "11:30 AM", "duration": "5h 30m"}'
The flight time from New York (NYC) to Los Angeles (LAX) is approximately 5 hours and 30 minutes. The departure time is at 8:00 AM, and the arrival time is at 11:30 AM.
"""
asyncio.run(run(
  model=model, query='User Jeffery is an admin of Ollama'
))
"""It should output something like:
function_response={'name': 'Jeffery', 'role': 'admin'}
The user Jeffery is indeed an admin of Ollama.
"""
