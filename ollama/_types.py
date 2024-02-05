import json
from enum import Enum
from typing import TypedDict, Sequence, Optional, Any

from pydantic import BaseModel, field_validator

from ollama._utils import _encode_image


class BaseGenerateResponse(BaseModel):
    model: str
    'Model used to generate response.'

    created_at: str
    'Time when the request was created.'

    done: bool
    'True if response is complete, otherwise False. Useful for streaming to detect the final response.'

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


class Role(str, Enum):
    USER = 'user'
    ASSISTANT = 'assistant'
    SYSTEM = 'system'


class Message(BaseModel):
    """
    Chat message.
    """

    role: Role
    "Assumed role of the message. Response messages always has role 'assistant'."

    content: str
    'Content of the message. Response messages contains message fragments when streaming.'

    images: Optional[list[str | bytes]] = None
    """
    Optional list of image data for multimodal models.
  
    Valid input types are:
  
    - `str` or path-like object: path to image file
    - `bytes` or bytes-like object: raw image data
  
    Valid image formats depend on the model. See the model card for more information.
    """

    @field_validator('images')
    @classmethod
    def encode_images(cls, v: Any):
        return [_encode_image(image) for image in v]


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


class Options(BaseModel):
    # load time options
    numa: bool
    num_ctx: int
    num_batch: int
    num_gqa: int
    num_gpu: int
    main_gpu: int
    low_vram: bool
    f16_kv: bool
    logits_all: bool
    vocab_only: bool
    use_mmap: bool
    use_mlock: bool
    embedding_only: bool
    rope_frequency_base: float
    rope_frequency_scale: float
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






