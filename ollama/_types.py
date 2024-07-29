import json
from typing import Any, TypedDict, Sequence, Literal, Mapping

import sys

if sys.version_info < (3, 11):
  from typing_extensions import NotRequired
else:
  from typing import NotRequired


class BaseGenerateResponse(TypedDict):
  model: str
  'Model used to generate response.'

  created_at: str
  'Time when the request was created.'

  done: bool
  'True if response is complete, otherwise False. Useful for streaming to detect the final response.'

  done_reason: str
  'Reason for completion. Only present when done is True.'

  total_duration: int
  'Total duration in nanoseconds.'

  load_duration: int
  'Load duration in nanoseconds.'

  prompt_eval_count: int
  'Number of tokens evaluated in the prompt.'

  prompt_eval_duration: int
  'Duration of evaluating the prompt in nanoseconds.'

  eval_count: int
  'Number of tokens evaluated in inference.'

  eval_duration: int
  'Duration of evaluating inference in nanoseconds.'


class GenerateResponse(BaseGenerateResponse):
  """
  Response returned by generate requests.
  """

  response: str
  'Response content. When streaming, this contains a fragment of the response.'

  context: Sequence[int]
  'Tokenized history up to the point of the response.'


class ToolCallFunction(TypedDict):
  """
  Tool call function.
  """

  name: str
  'Name of the function.'

  arguments: NotRequired[Mapping[str, Any]]
  'Arguments of the function.'


class ToolCall(TypedDict):
  """
  Model tool calls.
  """

  function: ToolCallFunction
  'Function to be called.'


class Message(TypedDict):
  """
  Chat message.
  """

  role: Literal['user', 'assistant', 'system', 'tool']
  "Assumed role of the message. Response messages always has role 'assistant' or 'tool'."

  content: NotRequired[str]
  'Content of the message. Response messages contains message fragments when streaming.'

  images: NotRequired[Sequence[Any]]
  """
  Optional list of image data for multimodal models.

  Valid input types are:

  - `str` or path-like object: path to image file
  - `bytes` or bytes-like object: raw image data

  Valid image formats depend on the model. See the model card for more information.
  """

  tool_calls: NotRequired[Sequence[ToolCall]]
  """
  Tools calls to be made by the model.
  """


class Property(TypedDict):
  type: str
  description: str
  enum: NotRequired[Sequence[str]]  # `enum` is optional and can be a list of strings


class Parameters(TypedDict):
  type: str
  required: Sequence[str]
  properties: Mapping[str, Property]


class ToolFunction(TypedDict):
  name: str
  description: str
  parameters: Parameters


class Tool(TypedDict):
  type: str
  function: ToolFunction


class ChatResponse(BaseGenerateResponse):
  """
  Response returned by chat requests.
  """

  message: Message
  'Response message.'


class ProgressResponse(TypedDict):
  status: str
  completed: int
  total: int
  digest: str


class Options(TypedDict, total=False):
  # load time options
  numa: bool
  num_ctx: int
  num_batch: int
  num_gpu: int
  main_gpu: int
  low_vram: bool
  f16_kv: bool
  logits_all: bool
  vocab_only: bool
  use_mmap: bool
  use_mlock: bool
  embedding_only: bool
  num_thread: int

  # runtime options
  num_keep: int
  seed: int
  num_predict: int
  top_k: int
  top_p: float
  tfs_z: float
  typical_p: float
  repeat_last_n: int
  temperature: float
  repeat_penalty: float
  presence_penalty: float
  frequency_penalty: float
  mirostat: int
  mirostat_tau: float
  mirostat_eta: float
  penalize_newline: bool
  stop: Sequence[str]


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
