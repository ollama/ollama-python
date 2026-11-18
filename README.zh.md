# Ollama Python 客户端库

**简体中文** | [**English**](README.md)

Ollama Python 库为 Python 3.8+ 项目提供了与 [Ollama](https://github.com/ollama/ollama) 进行深度集成的极简途径。

## 前置准备 (Prerequisites)

- 确保已安装并正在运行 [Ollama 服务](https://ollama.com/download)
- 拉取待使用的模型：`ollama pull <model>`，例如 `ollama pull gemma4`
  - 访问 [Ollama.com 官方模型库](https://ollama.com/search) 了解更多可用模型。

## 安装指南 (Install)

```sh
pip install ollama
```

## 基础用法 (Usage)

```python
from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(
  model='gemma4',
  messages=[
    {
      'role': 'user',
      'content': '天空为什么是蓝色的？',
    },
  ],
)
print(response['message']['content'])
# 亦可直接通过响应对象的属性访问字段
print(response.message.content)
```

关于响应对象的完整类型定义，请参阅 [_types.py](ollama/_types.py)。

## 流式响应输出 (Streaming responses)

通过设置 `stream=True` 可开启流式输出：

```python
from ollama import chat

stream = chat(
  model='gemma4',
  messages=[{'role': 'user', 'content': '天空为什么是蓝色的？'}],
  stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
```

## 云端模型与云推理 (Cloud Models)

通过将推理任务卸载至 Ollama 云端，既可运行更大参数规模的模型，又无需改变本地开发工作流。

- 支持的模型列表：请参阅[云端模型库](https://ollama.com/search?c=cloud)查看当前最新支持情况。

### 方式一：通过本地 Ollama 调度运行

1) 登录账户（仅需单次配置）：

```
ollama signin
```

2) 拉取云端模型：

```
ollama pull gpt-oss:120b-cloud
```

3) 发起对话请求：

```python
from ollama import Client

client = Client()

messages = [
  {
    'role': 'user',
    'content': '天空为什么是蓝色的？',
  },
]

for part in client.chat('gpt-oss:120b-cloud', messages=messages, stream=True):
  print(part.message.content, end='', flush=True)
```

### 方式二：云端直接 API 调用 (ollama.com)

将客户端直接指向 `https://ollama.com`，直接访问云端模型端点。

1) 在 [ollama.com/settings/keys](https://ollama.com/settings/keys) 创建 API 密钥，并配置环境变量：

```
export OLLAMA_API_KEY=your_api_key
```

2)（可选）查看云端 API 支持的模型列表：

```
curl https://ollama.com/api/tags
```

3) 通过云端 API 生成响应：

```python
import os
from ollama import Client

client = Client(host='https://ollama.com', headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')})

messages = [
  {
    'role': 'user',
    'content': '天空为什么是蓝色的？',
  },
]

for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
  print(part.message.content, end='', flush=True)
```

## 自定义客户端配置 (Custom client)

可以通过实例化 `ollama` 模块中的 `Client` 或 `AsyncClient` 创建自定义客户端。

所有额外的关键字参数都将透传至底层的 [`httpx.Client`](https://www.python-httpx.org/api/#client)。

```python
from ollama import Client

client = Client(host='http://localhost:11434', headers={'x-some-header': 'some-value'})
response = client.chat(
  model='gemma4',
  messages=[
    {
      'role': 'user',
      'content': '天空为什么是蓝色的？',
    },
  ],
)
```

## 异步客户端 (Async client)

`AsyncClient` 类用于发起异步非阻塞请求。其配置字段与 `Client` 完全一致。

```python
import asyncio
from ollama import AsyncClient


async def chat():
  message = {'role': 'user', 'content': '天空为什么是蓝色的？'}
  response = await AsyncClient().chat(model='gemma4', messages=[message])


asyncio.run(chat())
```

指定 `stream=True` 时，方法将返回一个 Python 异步生成器（Async Generator）：

```python
import asyncio
from ollama import AsyncClient


async def chat():
  message = {'role': 'user', 'content': '天空为什么是蓝色的？'}
  async for part in await AsyncClient().chat(model='gemma4', messages=[message], stream=True):
    print(part['message']['content'], end='', flush=True)


asyncio.run(chat())
```

## API 核心方法矩阵 (API)

Ollama Python 库的 API 设计紧密围绕 [Ollama REST API 规范](https://github.com/ollama/ollama/blob/main/docs/api.md)展开：

### 对话 (Chat)

```python
ollama.chat(model='gemma4', messages=[{'role': 'user', 'content': '天空为什么是蓝色的？'}])
```

### 文本补全与生成 (Generate)

```python
ollama.generate(model='gemma4', prompt='天空为什么是蓝色的？')
```

### 获取本地模型列表 (List)

```python
ollama.list()
```

### 查看模型详细元信息 (Show)

```python
ollama.show('gemma4')
```

### 创建自定义模型 (Create)

```python
ollama.create(model='example', from_='gemma4', system='你是来自超级马力欧兄弟里的马力欧。')
```

### 复制模型 (Copy)

```python
ollama.copy('gemma4', 'user/gemma4')
```

### 删除模型 (Delete)

```python
ollama.delete('gemma4')
```

### 拉取模型 (Pull)

```python
ollama.pull('gemma4')
```

### 推送模型至注册表 (Push)

```python
ollama.push('user/gemma4')
```

### 向量嵌入 (Embed)

```python
ollama.embed(model='gemma4', input='天空呈蓝色是因为瑞利散射效应')
```

### 批量向量嵌入 (Embed - batch)

```python
ollama.embed(model='gemma4', input=['天空呈蓝色是因为瑞利散射效应', '草地呈绿色是因为叶绿素的存在'])
```

### 查看运行中模型进程 (Ps)

```python
ollama.ps()
```

### 决策系统 (System One)

```python
import ollama

response = ollama.systemone(
  model='nimble',
  state='结账系统自早上 9 点起持续返回 500 错误。',
  questions={
    'team': {
      'type': 'choice',
      'instructions': '该工单应由哪个团队处理？',
      'criteria': {'billing': '支付与退款问题', 'technical': '软件服务与代码错误'},
    },
  },
)
print(response.answers['team'])
```

System One 底层调用 `POST /v1/systemone` 并返回单个 JSON 响应；该接口不支持流式输出。它可在本地使用专用决策模型运行，例如 `nimble`（要求 Ollama v0.35.0+）或 `clef`（要求 v0.35.1+），亦可在云端通过付费计划使用 `clef:cloud`（要求 Ollama v0.40.5+，或[直接调用 ollama.com 云端端点](#方式二云端直接-api-调用-ollamacom)）。

Clef 模型同样支持读取图像输入（本地需 v0.40.0+）：可将 `images` 作为文件路径、字节流或 Base64 编码字符串传入，供所有问题共享。云端请求最多接受 4 张 PNG、JPEG 或 WebP 格式图像（单张最大 4 MiB 且不超过 1600 万像素，总计最大 8 MiB）。

`state` 与各问题的 `instructions` 支持文本、JSON 对象或数组格式。各个问题将严格按照传入的顺序依次求值：

- `choice`：2 至 26 个选项键映射至对应描述说明（云端最高支持 255 个选项）；若设为 `None`，则直接将键名作为其描述。
- `noul`：求值为 True 的概率值，支持可选的 `{"false": "否", "true": "是"}` 标签描述。
- `score`：2 至 10 个按由低到高顺序排列的描述项；返回一个基于 0 起始的可能带小数的评分值。

在云端请求中，问题 ID 仅可包含 ASCII 英文字母、数字、`_`、`.` 和 `-`，长度不超过 100 个字符；单次请求最多可同时提问 64 个问题。

响应对象包含 `model`、`answers` 以及 `usage.input_tokens` / `usage.output_tokens`。置信度度量的是概率集中度，而非校准后的准确性。Token 消耗量由服务端返回；输出 Token 数不一定为零。可选的 `keep_alive` 参数接受秒数或时间字符串。所有请求均复用客户端既有的主机地址、请求头及 HTTP 异常处理机制。服务端会对请求体与模型上下文长度上限进行严格校验，不会静默截断输入内容。

该接口可通过 `ollama.systemone`、`Client.systemone` 以及 `await AsyncClient.systemone` 调用。包括 `SystemOneResponse` 在内的问题与解答类型均可直接从 `ollama` 顶层导出。完整的多问题综合调用范例可参阅 [examples/systemone.py](examples/systemone.py)。

## 异常与错误处理 (Errors)

当请求返回错误状态码或在流式传输过程中检测到异常时，将抛出错误对象：

```python
model = 'does-not-yet-exist'

try:
  ollama.chat(model)
except ollama.ResponseError as e:
  print('错误信息:', e.error)
  if e.status_code == 404:
    ollama.pull(model)
```
