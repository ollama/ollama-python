from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ollama import chat


# Define the schema for image objects
class Object(BaseModel):
  name: str
  confidence: float
  attributes: str


class ImageDescription(BaseModel):
  summary: str
  objects: list[Object]
  scene: str
  colors: list[str]
  time_of_day: Literal['Morning', 'Afternoon', 'Evening', 'Night']
  setting: Literal['Indoor', 'Outdoor', 'Unknown']
  text_content: str | None = None


# Get path from user input
path = input('Enter the path to your image: ')
path = Path(path)

# Verify the file exists
if not path.exists():
  raise FileNotFoundError(f'Image not found at: {path}')

# Set up chat as usual
response = chat(
  model='gemma3',
  format=ImageDescription.model_json_schema(),  # Pass in the schema for the response
  messages=[
    {
      'role': 'user',
      'content': 'Analyze this image and return a detailed JSON description including objects, scene, colors and any text detected. If you cannot determine certain details, leave those fields empty.',
      'images': [path],
    },
  ],
  options={'temperature': 0},  # Set temperature to 0 for more deterministic output
)


# Convert received content to the schema
image_analysis = ImageDescription.model_validate_json(response.message.content)
print(image_analysis)
