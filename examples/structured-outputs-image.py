from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional, Literal
from ollama import chat
from rich import print


# Define the schema for image objects
class Object(BaseModel):
  name: str
  confidence: float
  attributes: Optional[dict] = None


class ImageDescription(BaseModel):
  summary: str
  objects: List[Object]
  scene: str
  colors: List[str]
  time_of_day: Literal['Morning', 'Afternoon', 'Evening', 'Night']
  setting: Literal['Indoor', 'Outdoor', 'Unknown']
  text_content: Optional[str] = None


# Get path from user input
path = input('Enter the path to your image: ')
path = Path(path)

# Verify the file exists
if not path.exists():
  raise FileNotFoundError(f'Image not found at: {path}')

# Set up chat as usual
response = chat(
  model='llama3.2-vision',
  format=ImageDescription,  # Pass in the schema for the response
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
