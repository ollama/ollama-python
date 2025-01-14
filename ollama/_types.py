from __future__ import annotations

import json
from base64 import b64decode, b64encode
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import (
  BaseModel,
  ByteSize,
  ConfigDict,
  Field,
  model_serializer,
)
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import Annotated, Literal


class SubscriptableBaseModel(BaseModel):
  def __getitem__(self, key: str) -> Any:
    """
    >>> msg = Message(role='user')
    >>> msg['role']
    'user'
    >>> msg = Message(role='user')
    >>> msg['nonexistent']
    Traceback (most recent call last):
    KeyError: 'nonexistent'
    """
    if key in self:
      return getattr(self, key)

    raise KeyError(key)

  def __setitem__(self, key: str, value: Any) -> None:
    """
    >>> msg = Message(role='user')
    >>> msg['role'] = 'assistant'
    >>> msg['role']
    'assistant'
    >>> tool_call = Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))
    >>> msg = Message(role='user', content='hello')
    >>> msg['tool_calls'] = [tool_call]
    >>> msg['tool_calls'][0]['function']['name']
    'foo'
    """
    setattr(self, key, value)

  def __contains__(self, key: str) -> bool:
    """
    >>> msg = Message(role='user')
    >>> 'nonexistent' in msg
    False
    >>> 'role' in msg
    True
    >>> 'content' in msg
    False
    >>> msg.content = 'hello!'
    >>> 'content' in msg
    True
    >>> msg = Message(role='user', content='hello!')
    >>> 'content' in msg
    True
    >>> 'tool_calls' in msg
    False
    >>> msg['tool_calls'] = []
    >>> 'tool_calls' in msg
    True
    >>> msg['tool_calls'] = [Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))]
    >>> 'tool_calls' in msg
    True
    >>> msg['tool_calls'] = None
    >>> 'tool_calls' in msg
    True
    >>> tool = Tool()
    >>> 'type' in tool
    True
    """
    if key in self.model_fields_set:
      return True

    if key in self.model_fields:
      return self.model_fields[key].default is not None

    return False

  def get(self, key: str, default: Any = None) -> Any:
    """
    >>> msg = Message(role='user')
    >>> msg.get('role')
    'user'
    >>> msg = Message(role='user')
    >>> msg.get('nonexistent')
    >>> msg = Message(role='user')
    >>> msg.get('nonexistent', 'default')
    'default'
    >>> msg = Message(role='user', tool_calls=[ Message.ToolCall(function=Message.ToolCall.Function(name='foo', arguments={}))])
    >>> msg.get('tool_calls')[0]['function']['name']
    'foo'
    """
    return self[key] if key in self else default


class Options(SubscriptableBaseModel):
  # load time options
  numa: bool | None = None
  num_ctx: int | None = None
  num_batch: int | None = None
  num_gpu: int | None = None
  main_gpu: int | None = None
  low_vram: bool | None = None
  f16_kv: bool | None = None
  logits_all: bool | None = None
  vocab_only: bool | None = None
  use_mmap: bool | None = None
  use_mlock: bool | None = None
  embedding_only: bool | None = None
  num_thread: int | None = None

  # runtime options
  num_keep: int | None = None
  seed: int | None = None
  num_predict: int | None = None
  top_k: int | None = None
  top_p: float | None = None
  tfs_z: float | None = None
  typical_p: float | None = None
  repeat_last_n: int | None = None
  temperature: float | None = None
  repeat_penalty: float | None = None
  presence_penalty: float | None = None
  frequency_penalty: float | None = None
  mirostat: int | None = None
  mirostat_tau: float | None = None
  mirostat_eta: float | None = None
  penalize_newline: bool | None = None
  stop: Sequence[str] | None = None


class BaseRequest(SubscriptableBaseModel):
  model: Annotated[str, Field(min_length=1)]
  'Model to use for the request.'


class BaseStreamableRequest(BaseRequest):
  stream: bool | None = None
  'Stream response.'


class BaseGenerateRequest(BaseStreamableRequest):
  options: Mapping[str, Any] | Options | None = None
  'Options to use for the request.'

  format: Literal['', 'json'] | JsonSchemaValue | None = None
  'Format of the response.'

  keep_alive: float | str | None = None
  'Keep model alive for the specified duration.'


class Image(BaseModel):
  value: str | bytes | Path

  @model_serializer
  def serialize_model(self):
    if isinstance(self.value, (Path, bytes)):
      return b64encode(self.value.read_bytes() if isinstance(self.value, Path) else self.value).decode()

    if isinstance(self.value, str):
      try:
        if Path(self.value).exists():
          return b64encode(Path(self.value).read_bytes()).decode()
      except Exception:
        # Long base64 string can't be wrapped in Path, so try to treat as base64 string
        pass

      # String might be a file path, but might not exist
      if self.value.split('.')[-1] in ('png', 'jpg', 'jpeg', 'webp'):
        raise ValueError(f'File {self.value} does not exist')

      try:
        # Try to decode to check if it's already base64
        b64decode(self.value)
        return self.value
      except Exception:
        raise ValueError('Invalid image data, expected base64 string or path to image file') from Exception


class GenerateRequest(BaseGenerateRequest):
  prompt: str | None = None
  'Prompt to generate response from.'

  suffix: str | None = None
  'Suffix to append to the response.'

  system: str | None = None
  'System prompt to prepend to the prompt.'

  template: str | None = None
  'Template to use for the response.'

  context: Sequence[int] | None = None
  'Tokenized history to use for the response.'

  raw: bool | None = None

  images: Sequence[Image] | None = None
  'Image data for multimodal models.'


class BaseGenerateResponse(SubscriptableBaseModel):
  model: str | None = None
  'Model used to generate response.'

  created_at: str | None = None
  'Time when the request was created.'

  done: bool | None = None
  'True if response is complete, otherwise False. Useful for streaming to detect the final response.'

  done_reason: str | None = None
  'Reason for completion. Only present when done is True.'

  total_duration: int | None = None
  'Total duration in nanoseconds.'

  load_duration: int | None = None
  'Load duration in nanoseconds.'

  prompt_eval_count: int | None = None
  'Number of tokens evaluated in the prompt.'

  prompt_eval_duration: int | None = None
  'Duration of evaluating the prompt in nanoseconds.'

  eval_count: int | None = None
  'Number of tokens evaluated in inference.'

  eval_duration: int | None = None
  'Duration of evaluating inference in nanoseconds.'


class GenerateResponse(BaseGenerateResponse):
  """
  Response returned by generate requests.
  """

  response: str
  'Response content. When streaming, this contains a fragment of the response.'

  context: Sequence[int] | None = None
  'Tokenized history up to the point of the response.'


class Message(SubscriptableBaseModel):
  """
  Chat message.
  """

  role: Literal['user', 'assistant', 'system', 'tool']
  "Assumed role of the message. Response messages has role 'assistant' or 'tool'."

  content: str | None = None
  'Content of the message. Response messages contains message fragments when streaming.'

  images: Sequence[Image] | None = None
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

  tool_calls: Sequence[ToolCall] | None = None
  """
  Tools calls to be made by the model.
  """


class Tool(SubscriptableBaseModel):
  type: Literal['function'] | None = 'function'

  class Function(SubscriptableBaseModel):
    name: str | None = None
    description: str | None = None

    class Parameters(SubscriptableBaseModel):
      type: Literal['object'] | None = 'object'
      required: Sequence[str] | None = None

      class Property(SubscriptableBaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        type: str | None = None
        description: str | None = None

      properties: Mapping[str, Property] | None = None

    parameters: Parameters | None = None

  function: Function | None = None


class ChatRequest(BaseGenerateRequest):
  messages: Sequence[Mapping[str, Any] | Message] | None = None
  'Messages to chat with.'

  tools: Sequence[Tool] | None = None
  'Tools to use for the chat.'


class ChatResponse(BaseGenerateResponse):
  """
  Response returned by chat requests.
  """

  message: Message
  'Response message.'


class EmbedRequest(BaseRequest):
  input: str | Sequence[str]
  'Input text to embed.'

  truncate: bool | None = None
  'Truncate the input to the maximum token length.'

  options: Mapping[str, Any] | Options | None = None
  'Options to use for the request.'

  keep_alive: float | str | None = None


class EmbedResponse(BaseGenerateResponse):
  """
  Response returned by embed requests.
  """

  embeddings: Sequence[Sequence[float]]
  'Embeddings of the inputs.'


class EmbeddingsRequest(BaseRequest):
  prompt: str | None = None
  'Prompt to generate embeddings from.'

  options: Mapping[str, Any] | Options | None = None
  'Options to use for the request.'

  keep_alive: float | str | None = None


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

  insecure: bool | None = None
  'Allow insecure (HTTP) connections.'


class PushRequest(BaseStreamableRequest):
  """
  Request to pull the model.
  """

  insecure: bool | None = None
  'Allow insecure (HTTP) connections.'


class CreateRequest(BaseStreamableRequest):
  @model_serializer(mode='wrap')
  def serialize_model(self, nxt):
    output = nxt(self)
    if 'from_' in output:
      output['from'] = output.pop('from_')
    return output

  """
  Request to create a new model.
  """
  quantize: str | None = None
  from_: str | None = None
  files: dict[str, str] | None = None
  adapters: dict[str, str] | None = None
  template: str | None = None
  license: str | list[str] | None = None
  system: str | None = None
  parameters: Mapping[str, Any] | Options | None = None
  messages: Sequence[Mapping[str, Any] | Message] | None = None


class ModelDetails(SubscriptableBaseModel):
  parent_model: str | None = None
  format: str | None = None
  family: str | None = None
  families: Sequence[str] | None = None
  parameter_size: str | None = None
  quantization_level: str | None = None


class ListResponse(SubscriptableBaseModel):
  class Model(SubscriptableBaseModel):
    model: str | None = None
    modified_at: datetime | None = None
    digest: str | None = None
    size: ByteSize | None = None
    details: ModelDetails | None = None

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
  status: str | None = None


class ProgressResponse(StatusResponse):
  completed: int | None = None
  total: int | None = None
  digest: str | None = None


class ShowRequest(BaseRequest):
  """
  Request to show model information.
  """


class ShowResponse(SubscriptableBaseModel):
  modified_at: datetime | None = None

  template: str | None = None

  modelfile: str | None = None

  license: str | None = None

  details: ModelDetails | None = None

  modelinfo: Mapping[str, Any] | None = Field(alias='model_info')

  parameters: str | None = None


class ProcessResponse(SubscriptableBaseModel):
  class Model(SubscriptableBaseModel):
    model: str | None = None
    name: str | None = None
    digest: str | None = None
    expires_at: datetime | None = None
    size: ByteSize | None = None
    size_vram: ByteSize | None = None
    details: ModelDetails | None = None

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

  def __str__(self) -> str:
    return f'{self.error} (status code: {self.status_code})'
