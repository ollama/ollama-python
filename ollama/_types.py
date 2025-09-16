import contextlib
import json
from base64 import b64decode, b64encode
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

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

    if value := self.__class__.model_fields.get(key):
      return value.default is not None

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
    return getattr(self, key) if hasattr(self, key) else default


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

  format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None
  'Format of the response.'

  keep_alive: Optional[Union[float, str]] = None
  'Keep model alive for the specified duration.'


class Image(BaseModel):
  value: Union[str, bytes, Path]

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

  think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None
  'Enable thinking mode (for thinking models).'


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

  thinking: Optional[str] = None
  'Thinking content. Only present when thinking is enabled.'

  context: Optional[Sequence[int]] = None
  'Tokenized history up to the point of the response.'


class Message(SubscriptableBaseModel):
  """
  Chat message.
  """

  role: str
  "Assumed role of the message. Response messages has role 'assistant' or 'tool'."

  content: Optional[str] = None
  'Content of the message. Response messages contains message fragments when streaming.'

  thinking: Optional[str] = None
  'Thinking content. Only present when thinking is enabled.'

  images: Optional[Sequence[Image]] = None
  """
  Optional list of image data for multimodal models.

  Valid input types are:

  - `str` or path-like object: path to image file
  - `bytes` or bytes-like object: raw image data

  Valid image formats depend on the model. See the model card for more information.
  """

  tool_name: Optional[str] = None
  'Name of the executed tool.'

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
  type: Optional[str] = 'function'

  class Function(SubscriptableBaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

    class Parameters(SubscriptableBaseModel):
      model_config = ConfigDict(populate_by_name=True)
      type: Optional[Literal['object']] = 'object'
      defs: Optional[Any] = Field(None, alias='$defs')
      items: Optional[Any] = None
      required: Optional[Sequence[str]] = None

      class Property(SubscriptableBaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        type: Optional[Union[str, Sequence[str]]] = None
        items: Optional[Any] = None
        description: Optional[str] = None
        enum: Optional[Sequence[Any]] = None

      properties: Optional[Mapping[str, Property]] = None

    parameters: Optional[Parameters] = None

  function: Optional[Function] = None


class ChatRequest(BaseGenerateRequest):
  @model_serializer(mode='wrap')
  def serialize_model(self, nxt):
    output = nxt(self)
    if output.get('tools'):
      for tool in output['tools']:
        if 'function' in tool and 'parameters' in tool['function'] and 'defs' in tool['function']['parameters']:
          tool['function']['parameters']['$defs'] = tool['function']['parameters'].pop('defs')
    return output

  messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None
  'Messages to chat with.'

  tools: Optional[Sequence[Tool]] = None
  'Tools to use for the chat.'

  think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None
  'Enable thinking mode (for thinking models).'


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

  dimensions: Optional[int] = None
  'Dimensions truncates the output embedding to the specified dimension.'


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
  @model_serializer(mode='wrap')
  def serialize_model(self, nxt):
    output = nxt(self)
    if 'from_' in output:
      output['from'] = output.pop('from_')
    return output

  """
  Request to create a new model.
  """
  quantize: Optional[str] = None
  from_: Optional[str] = None
  files: Optional[Dict[str, str]] = None
  adapters: Optional[Dict[str, str]] = None
  template: Optional[str] = None
  license: Optional[Union[str, List[str]]] = None
  system: Optional[str] = None
  parameters: Optional[Union[Mapping[str, Any], Options]] = None
  messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None


class ModelDetails(SubscriptableBaseModel):
  parent_model: Optional[str] = None
  format: Optional[str] = None
  family: Optional[str] = None
  families: Optional[Sequence[str]] = None
  parameter_size: Optional[str] = None
  quantization_level: Optional[str] = None


class ListResponse(SubscriptableBaseModel):
  class Model(SubscriptableBaseModel):
    model: Optional[str] = None
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

  capabilities: Optional[List[str]] = None


class ProcessResponse(SubscriptableBaseModel):
  class Model(SubscriptableBaseModel):
    model: Optional[str] = None
    name: Optional[str] = None
    digest: Optional[str] = None
    expires_at: Optional[datetime] = None
    size: Optional[ByteSize] = None
    size_vram: Optional[ByteSize] = None
    details: Optional[ModelDetails] = None
    context_length: Optional[int] = None

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
    # try to parse content as JSON and extract 'error'
    # fallback to raw content if JSON parsing fails
    with contextlib.suppress(json.JSONDecodeError):
      error = json.loads(error).get('error', error)

    super().__init__(error)
    self.error = error
    'Reason for the error.'

    self.status_code = status_code
    'HTTP status code of the response.'

  def __str__(self) -> str:
    return f'{self.error} (status code: {self.status_code})'
