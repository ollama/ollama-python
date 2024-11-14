import json
from base64 import b64encode
from pathlib import Path
from datetime import datetime
from typing import Any, List, Mapping, Optional, TypeVar, Union, Sequence, get_args, get_origin
from collections.abc import Set
from typing import Dict, Set as TypeSet

import sys
from typing_extensions import Annotated, Literal

from pydantic import (
  BaseModel,
  ByteSize,
  ConfigDict,
  Field,
  FilePath,
  Base64Str,
  model_serializer,
)

T = TypeVar('T')
if sys.version_info >= (3, 10):
  from types import UnionType
else:
  UnionType = Union[T]


class SubscriptableBaseModel(BaseModel):
  def __getitem__(self, key: str) -> Any:
    return getattr(self, key)

  def __setitem__(self, key: str, value: Any) -> None:
    setattr(self, key, value)

  def __contains__(self, key: str) -> bool:
    return hasattr(self, key)

  def get(self, key: str, default: Any = None) -> Any:
    return getattr(self, key, default)


class Options(SubscriptableBaseModel):
  # load time options
  numa: Optional[bool] = None
  num_ctx: Optional[int] = None
  num_batch: Optional[int] = None
  num_gpu: Optional[int] = None
  main_gpu: Optional[int] = None
  low_vram: Optional[bool] = None
  f16_kv: Optional[bool] = None
  logits_all: Optional[bool] = None
  vocab_only: Optional[bool] = None
  use_mmap: Optional[bool] = None
  use_mlock: Optional[bool] = None
  embedding_only: Optional[bool] = None
  num_thread: Optional[int] = None

  # runtime options
  num_keep: Optional[int] = None
  seed: Optional[int] = None
  num_predict: Optional[int] = None
  top_k: Optional[int] = None
  top_p: Optional[float] = None
  tfs_z: Optional[float] = None
  typical_p: Optional[float] = None
  repeat_last_n: Optional[int] = None
  temperature: Optional[float] = None
  repeat_penalty: Optional[float] = None
  presence_penalty: Optional[float] = None
  frequency_penalty: Optional[float] = None
  mirostat: Optional[int] = None
  mirostat_tau: Optional[float] = None
  mirostat_eta: Optional[float] = None
  penalize_newline: Optional[bool] = None
  stop: Optional[Sequence[str]] = None


class BaseRequest(SubscriptableBaseModel):
  model: Annotated[str, Field(min_length=1)]
  'Model to use for the request.'


class BaseStreamableRequest(BaseRequest):
  stream: Optional[bool] = None
  'Stream response.'


class BaseGenerateRequest(BaseStreamableRequest):
  options: Optional[Union[Mapping[str, Any], Options]] = None
  'Options to use for the request.'

  format: Optional[Literal['', 'json']] = None
  'Format of the response.'

  keep_alive: Optional[Union[float, str]] = None
  'Keep model alive for the specified duration.'


class Image(BaseModel):
  value: Union[FilePath, Base64Str, bytes]

  # This overloads the `model_dump` method and returns values depending on the type of the `value` field
  @model_serializer
  def serialize_model(self):
    if isinstance(self.value, Path):
      return b64encode(self.value.read_bytes()).decode()
    elif isinstance(self.value, bytes):
      return b64encode(self.value).decode()
    return self.value


class GenerateRequest(BaseGenerateRequest):
  prompt: Optional[str] = None
  'Prompt to generate response from.'

  suffix: Optional[str] = None
  'Suffix to append to the response.'

  system: Optional[str] = None
  'System prompt to prepend to the prompt.'

  template: Optional[str] = None
  'Template to use for the response.'

  context: Optional[Sequence[int]] = None
  'Tokenized history to use for the response.'

  raw: Optional[bool] = None

  images: Optional[Sequence[Image]] = None
  'Image data for multimodal models.'


class BaseGenerateResponse(SubscriptableBaseModel):
  model: Optional[str] = None
  'Model used to generate response.'

  created_at: Optional[str] = None
  'Time when the request was created.'

  done: Optional[bool] = None
  'True if response is complete, otherwise False. Useful for streaming to detect the final response.'

  done_reason: Optional[str] = None
  'Reason for completion. Only present when done is True.'

  total_duration: Optional[int] = None
  'Total duration in nanoseconds.'

  load_duration: Optional[int] = None
  'Load duration in nanoseconds.'

  prompt_eval_count: Optional[int] = None
  'Number of tokens evaluated in the prompt.'

  prompt_eval_duration: Optional[int] = None
  'Duration of evaluating the prompt in nanoseconds.'

  eval_count: Optional[int] = None
  'Number of tokens evaluated in inference.'

  eval_duration: Optional[int] = None
  'Duration of evaluating inference in nanoseconds.'


class GenerateResponse(BaseGenerateResponse):
  """
  Response returned by generate requests.
  """

  response: str
  'Response content. When streaming, this contains a fragment of the response.'

  context: Optional[Sequence[int]] = None
  'Tokenized history up to the point of the response.'


class Message(SubscriptableBaseModel):
  """
  Chat message.
  """

  role: Literal['user', 'assistant', 'system', 'tool']
  "Assumed role of the message. Response messages has role 'assistant' or 'tool'."

  content: Optional[str] = None
  'Content of the message. Response messages contains message fragments when streaming.'

  images: Optional[Sequence[Image]] = None
  """
  Optional list of image data for multimodal models.

  Valid input types are:

  - `str` or path-like object: path to image file
  - `bytes` or bytes-like object: raw image data

  Valid image formats depend on the model. See the model card for more information.
  """

  class ToolCall(SubscriptableBaseModel):
    """
    Model tool calls.
    """

    class Function(SubscriptableBaseModel):
      """
      Tool call function.
      """

      name: str
      'Name of the function.'

      arguments: Mapping[str, Any]
      'Arguments of the function.'

    function: Function
    'Function to be called.'

  tool_calls: Optional[Sequence[ToolCall]] = None
  """
  Tools calls to be made by the model.
  """


class Tool(SubscriptableBaseModel):
  type: Literal['function'] = 'function'

  class Function(SubscriptableBaseModel):
    name: str
    description: str

    class Parameters(SubscriptableBaseModel):
      type: str
      required: Optional[Sequence[str]] = None

      class Property(SubscriptableBaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        type: Union[type, UnionType, Optional[T], str]
        description: str

        @model_serializer
        def serialize_model(self) -> dict:
          return {'type': _get_json_type(self.type), 'description': self.description}

      properties: Optional[Mapping[str, Property]] = None

    parameters: Parameters

  function: Function


class ChatRequest(BaseGenerateRequest):
  messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None
  'Messages to chat with.'

  tools: Optional[Sequence[Tool]] = None
  'Tools to use for the chat.'


class ChatResponse(BaseGenerateResponse):
  """
  Response returned by chat requests.
  """

  message: Message
  'Response message.'


class EmbedRequest(BaseRequest):
  input: Union[str, Sequence[str]]
  'Input text to embed.'

  truncate: Optional[bool] = None
  'Truncate the input to the maximum token length.'

  options: Optional[Union[Mapping[str, Any], Options]] = None
  'Options to use for the request.'

  keep_alive: Optional[Union[float, str]] = None


class EmbedResponse(BaseGenerateResponse):
  """
  Response returned by embed requests.
  """

  embeddings: Sequence[Sequence[float]]
  'Embeddings of the inputs.'


class EmbeddingsRequest(BaseRequest):
  prompt: Optional[str] = None
  'Prompt to generate embeddings from.'

  options: Optional[Union[Mapping[str, Any], Options]] = None
  'Options to use for the request.'

  keep_alive: Optional[Union[float, str]] = None


class EmbeddingsResponse(SubscriptableBaseModel):
  """
  Response returned by embeddings requests.
  """

  embedding: Sequence[float]
  'Embedding of the prompt.'


class PullRequest(BaseStreamableRequest):
  """
  Request to pull the model.
  """

  insecure: Optional[bool] = None
  'Allow insecure (HTTP) connections.'


class PushRequest(BaseStreamableRequest):
  """
  Request to pull the model.
  """

  insecure: Optional[bool] = None
  'Allow insecure (HTTP) connections.'


class CreateRequest(BaseStreamableRequest):
  """
  Request to create a new model.
  """

  modelfile: Optional[str] = None

  quantize: Optional[str] = None


class ModelDetails(SubscriptableBaseModel):
  parent_model: Optional[str] = None
  format: Optional[str] = None
  family: Optional[str] = None
  families: Optional[Sequence[str]] = None
  parameter_size: Optional[str] = None
  quantization_level: Optional[str] = None


class ListResponse(SubscriptableBaseModel):
  class Model(SubscriptableBaseModel):
    name: Optional[str] = None
    modified_at: Optional[datetime] = None
    digest: Optional[str] = None
    size: Optional[ByteSize] = None
    details: Optional[ModelDetails] = None

  models: Sequence[Model]
  'List of models.'


class DeleteRequest(BaseRequest):
  """
  Request to delete a model.
  """


class CopyRequest(BaseModel):
  """
  Request to copy a model.
  """

  source: str
  'Source model to copy.'

  destination: str
  'Destination model to copy to.'


class StatusResponse(SubscriptableBaseModel):
  status: Optional[str] = None


class ProgressResponse(StatusResponse):
  completed: Optional[int] = None
  total: Optional[int] = None
  digest: Optional[str] = None


class ShowRequest(BaseRequest):
  """
  Request to show model information.
  """


class ShowResponse(SubscriptableBaseModel):
  modified_at: Optional[datetime] = None

  template: Optional[str] = None

  modelfile: Optional[str] = None

  license: Optional[str] = None

  details: Optional[ModelDetails] = None

  modelinfo: Optional[Mapping[str, Any]] = Field(alias='model_info')

  parameters: Optional[str] = None


class ProcessResponse(SubscriptableBaseModel):
  class Model(SubscriptableBaseModel):
    model: Optional[str] = None
    name: Optional[str] = None
    digest: Optional[str] = None
    expires_at: Optional[datetime] = None
    size: Optional[ByteSize] = None
    size_vram: Optional[ByteSize] = None
    details: Optional[ModelDetails] = None

  models: Sequence[Model]


class RequestError(Exception):
  """
  Common class for request errors.
  """

  def __init__(self, error: str):
    super().__init__(error)
    self.error = error
    'Reason for the error.'


class ResponseError(Exception):
  """
  Common class for response errors.
  """

  def __init__(self, error: str, status_code: int = -1):
    try:
      # try to parse content as JSON and extract 'error'
      # fallback to raw content if JSON parsing fails
      error = json.loads(error).get('error', error)
    except json.JSONDecodeError:
      ...

    super().__init__(error)
    self.error = error
    'Reason for the error.'

    self.status_code = status_code
    'HTTP status code of the response.'


# Python doesn't have a type serializer, so we need to map types to JSON types
TYPE_MAP = {
  # Basic types
  int: 'integer',
  'int': 'integer',
  'integer': 'integer',
  str: 'string',
  'str': 'string',
  'string': 'string',
  float: 'number',
  'float': 'number',
  'number': 'number',
  bool: 'boolean',
  'bool': 'boolean',
  'boolean': 'boolean',
  type(None): 'null',
  None: 'null',
  'None': 'null',
  'null': 'null',
  # Collection types
  list: 'array',
  'list': 'array',
  List: 'array',
  'List': 'array',
  Sequence: 'array',
  'Sequence': 'array',
  tuple: 'array',
  'tuple': 'array',
  set: 'array',
  'set': 'array',
  Set: 'array',
  TypeSet: 'array',
  'Set': 'array',
  'array': 'array',
  # Mapping types
  dict: 'object',
  'dict': 'object',
  Dict: 'object',
  'Dict': 'object',
  Mapping: 'object',
  'Mapping': 'object',
  'object': 'object',
  Any: 'string',
  'Any': 'string',
}

if sys.version_info >= (3, 10):
  from types import UnionType

  def is_union(tp: Any) -> bool:
    return get_origin(tp) in (Union, UnionType)
else:

  def is_union(tp: Any) -> bool:
    return get_origin(tp) is Union


def map_type(python_type: Any) -> str:
  # Handle generic types (List[int], Dict[str, int], etc.)
  origin = get_origin(python_type)
  if origin is not None:
    # Get the base type (List, Dict, etc.)
    base_type = TYPE_MAP.get(origin, None)
    if base_type:
      return base_type
    # If it's a subclass of known abstract base classes, map to appropriate type
    if isinstance(origin, type):
      if issubclass(origin, (list, Sequence, tuple, set, Set)):
        return 'array'
      if issubclass(origin, (dict, Mapping)):
        return 'object'

  # Handle both type objects and type references (older Python versions)
  type_key = python_type
  if isinstance(python_type, type):
    type_key = python_type
  elif isinstance(python_type, str):
    type_key = python_type

  # If type not found in map, try to get the type name
  if type_key not in TYPE_MAP and hasattr(python_type, '__name__'):
    type_key = python_type.__name__

  if type_key in TYPE_MAP:
    return TYPE_MAP[type_key]

  raise ValueError(f'Could not map Python type {python_type} to a valid JSON type')


def _get_json_type(python_type: Union[type, UnionType, Optional[T]]) -> Union[str, List[str]]:
  # Handle Optional types (Union[type, None] and type | None)
  if is_union(python_type):
    args = get_args(python_type)
    # Filter out None/NoneType from union args
    if non_none_args := [arg for arg in args if arg not in (None, type(None))]:
      if len(non_none_args) == 1:
        return map_type(non_none_args[0])
      # For multiple return types (e.g., int | str | None), return stringified array of types -> "['integer', 'string', 'null']"
      return str([map_type(arg) for arg in non_none_args]).replace(' ', '')
  return map_type(python_type)


def _is_optional_type(python_type: Any) -> bool:
  if is_union(python_type):
    args = get_args(python_type)
    return any(arg in (None, type(None)) for arg in args)
  return False
