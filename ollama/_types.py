from typing import Any, TypedDict, Sequence

import sys

if sys.version_info < (3, 11):
  from typing_extensions import NotRequired
else:
  from typing import NotRequired


class BaseGenerateResponse(TypedDict):
  model: str
  created_at: str
  done: bool

  total_duration: int
  load_duration: int
  prompt_eval_count: int
  prompt_eval_duration: int
  eval_count: int
  eval_duration: int


class GenerateResponse(BaseGenerateResponse):
  response: str
  context: Sequence[int]


class Message(TypedDict):
  role: str
  content: str
  images: NotRequired[Sequence[Any]]


class ChatResponse(BaseGenerateResponse):
  message: Message


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
