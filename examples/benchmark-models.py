import time
import ollama

# ---- PROMPT ----
PROMPT = "Explain Generative AI in 100 words."

# ---- Get local models ----
models = [m.model for m in ollama.list().models]

# ---- Header ----
print(
    f"{'model name':<20} | "
    f"{'total(s)':>10} | "
    f"{'ttft(s)':>10} | "
    f"{'tok/s':>10} | "
    f"{'tokens':>10}"
)
print("-" * 72)

# ---- Benchmark ----
for model in models:

    start = time.perf_counter()
    first_token = None
    token_count = 0

    for chunk in ollama.generate(model=model, prompt=PROMPT, stream=True):
        if first_token is None:
            first_token = time.perf_counter()
        token_count += 1

    end = time.perf_counter()

    total_time = end - start
    ttft = (first_token - start) if first_token else total_time
    tps = token_count / total_time if total_time > 0 else 0.0

    print(
        f"{model:<20} | "
        f"{total_time:>10.3f} | "
        f"{ttft:>10.3f} | "
        f"{tps:>10.2f} | "
        f"{token_count:>10}"
    )