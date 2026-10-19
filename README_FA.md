# کتابخانه پایتون Ollama - مستندات فارسی

کتابخانه رسمی پایتون برای اجرای مدل‌های زبانی بزرگ (LLM) به صورت کاملا لوکال و آفلاین با [Ollama](https://ollama.com).

[English README](./README.md)

این مستندات برای جامعه فارسی‌زبان و توسعه‌دهندگان ایرانی نوشته شده تا سریع‌ترین راه برای ساخت چت‌بات فارسی، پردازش فاکتور و RAG را داشته باشند.

## 1. نصب و راه‌اندازی

```bash
pip install ollama
ollama serve
ollama pull llama3.1:8b
```

## 2. شروع سریع - اولین چت فارسی

```python
import ollama

response = ollama.chat(model='llama3.1:8b', messages=[
  {
    'role': 'user',
    'content': 'سلام! یک شعر کوتاه درباره تهران بگو.'
  },
])

print(response['message']['content'])
```

## 3. چت استریم (برای ربات تلگرام و وب‌سایت)

```python
import ollama

stream = ollama.chat(
    model='llama3.1:8b',
    messages=[{'role': 'user', 'content': 'درباره هوش مصنوعی توضیح بده'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
```

## 4. چت‌بات با حافظه مکالمه

```python
import ollama

messages = []

while True:
  user_input = input("شما: ")
  if user_input == "خروج":
    break
  messages.append({'role': 'user', 'content': user_input})
  response = ollama.chat(model='llama3.1:8b', messages=messages)
  bot_reply = response['message']['content']
  print(f"ربات: {bot_reply}")
  messages.append({'role': 'assistant', 'content': bot_reply})
```

## 5. پردازش تصویر - خواندن فاکتور فارسی با LLaVA

```python
import ollama

response = ollama.chat(
    model='llava:7b',
    messages=[{
        'role': 'user',
        'content': 'این تصویر یک فاکتور فارسی است. مبلغ کل و تاریخ را استخراج کن.',
        'images': ['./factor.jpg']
    }]
)

print(response['message']['content'])
```

## 6. نسخه Async برای FastAPI

```python
import asyncio
from ollama import AsyncClient

async def main():
  client = AsyncClient()
  response = await client.chat(model='llama3.1:8b', messages=[
    {'role': 'user', 'content': 'سلام'}
  ])
  print(response['message']['content'])

asyncio.run(main())
```

## 7. راهنمای سخت‌افزار

| مدل | RAM لازم | VRAM لازم |
| :--- | :--- | :--- |
| llama3.1:8b | 8GB | 6GB |
| llava:7b | 8GB | 6GB |
| llama3.1:70b | 64GB | 40GB |

ساخته شده برای کامیونیتی برنامه‌نویسان فارسی
