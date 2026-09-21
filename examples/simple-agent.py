import os
import subprocess
import ollama

model = "qwen3:4b"

system = "You are a smart agent. Use bash to solve some tasks."

tools = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]


def run_bash(command: str) -> str:
    dangerous = ["rm -rf", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def agent_loop(messages: list):
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": system})

    while True:
        response = ollama.chat(
            model=model,
            messages=messages,
            tools=tools,
            options={"num_predict": 8000},
        )

        msg = response["message"]
        assistant_content = msg.get("content", "")
        assistant_msg = {"role": "assistant", "content": assistant_content}
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            return

        for tool in tool_calls:
            func_name = tool["function"]["name"]
            if func_name == "bash":
                import json
                args = tool["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                command = args["command"]
                output = run_bash(command)
                messages.append({
                    "role": "tool",
                    "content": output,
                    "tool_name": "bash",
                })


if __name__ == "__main__":
    try:
        ollama.show(model)
    except Exception:
        print(f"\033[31mModel '{model}' not found locally.\033[0m")
        print(f"Pull it first with: \033[33mollama pull {model}\033[0m")
        exit(1)

    history = []
    print(f"\033[32mOllama agent loop running with {model}\033[0m")
    print("Type 'q' or 'exit' to quit.\n")

    while True:
        try:
            query = input("\033[36mOllama-Agent >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for entry in reversed(history):
            if entry.get("role") == "assistant" and entry.get("content"):
                print(entry["content"])
                break
        print()
