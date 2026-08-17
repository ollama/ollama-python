from ollama import Client

# num_gpu controls how many model layers are offloaded to the GPU for this request.
client = Client()
response = client.generate('gemma3', 'Why is the sky blue?', num_gpu=1)
print(response['response'])

# For hard isolation across multiple GPUs (e.g. pinning separate processes to
# different physical GPUs on a shared server), run one `ollama serve` per GPU
# with its own CUDA_VISIBLE_DEVICES and point a separate Client(host=...) at each:
#
#   CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 ollama serve
#   CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 ollama serve
#
# gpu0_client = Client(host='http://127.0.0.1:11434')
# gpu1_client = Client(host='http://127.0.0.1:11435')
