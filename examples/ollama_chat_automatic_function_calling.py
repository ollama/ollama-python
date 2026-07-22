from ollama import ChatResponse, chat


def ollama_automatic_function_calling(
    client_fn: chat,
    messages: list[dict],
    options: dict = None,
    **kwargs,
) -> tuple[ChatResponse, list[dict]]:
    """
    Automatically handles tool/function calls in a chat loop.

    max_turns is reached.

    Args:
        client_fn: Ollama chat client
        messages: List of message dicts for the conversation
        max_turns: Maximum number of chat iterations (default: 20)
        model: Model name to use (default: "glm-5:cloud")
        tools: List of available tool definitions
        tool_map: Dict mapping tool names to callable functions
        chat_fn: Optional callable to use for chat (default: ollama.chat)

    Returns:
        (last_response, messages)
    """
    model = kwargs.get("model")
    if model is None:
        raise ValueError("model must be specified")

    max_turns = kwargs.get("max_turns", 20)
    tools = kwargs.get("tools")
    tool_map = kwargs.get("tool_map")
    last_response = None

    def _to_msg_dict(msg):
        if hasattr(msg, "model_dump"):
            return msg.model_dump(exclude_none=True)
        if isinstance(msg, dict):
            return msg
        return {
            "role": getattr(msg, "role", "assistant"),
            "content": getattr(msg, "content", ""),
        }

    def _extract_tool_call_parts(tool_call):
        function = getattr(tool_call, "function", None)
        if function is None and isinstance(tool_call, dict):
            function = tool_call.get("function", {})

        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments") or {}
        else:
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None) or {}

        return name, arguments

    def _infer_tool_name(t):
        if t is None:
            return None
        if isinstance(t, dict):
            fn = t.get("function") or {}
            if isinstance(fn, dict):
                return fn.get("name")
            return None
        return getattr(t, "name", None) or getattr(t, "__name__", None)

    def _resolve_callable(fn):
        wrapped = getattr(fn, "__wrapped__", None)
        return wrapped if callable(wrapped) else fn

    if tool_map is None and tools:
        tool_map = {}
        for t in tools:
            name = _infer_tool_name(t)
            if name:
                tool_map[name] = t

    if not callable(client_fn):
        raise ValueError("client_fn must be a callable")

    reserved = {"model", "tools", "messages", "options", "max_turns", "tool_map"}
    chat_kwargs = {k: v for k, v in kwargs.items() if k not in reserved}

    for _ in range(max_turns):
        response = client_fn(
            model=model,
            tools=tools,
            messages=messages,
            options=options,
            **chat_kwargs,
        )
        last_response = response
        assistant_msg = response.message
        messages.append(_to_msg_dict(assistant_msg))

        tool_calls = getattr(assistant_msg, "tool_calls", None) or []
        if not tool_calls:
            return last_response, messages

        for tool_call in tool_calls:
            tool_name, tool_args = _extract_tool_call_parts(tool_call)
            tool_fn = tool_map.get(tool_name) if tool_map else None

            if not callable(tool_fn):
                messages.append(
                    {
                        "role": "tool",
                        "content": f"Tool '{tool_name}' not found.",
                        "tool_name": tool_name or "unknown_tool",
                    }
                )
                continue

            if isinstance(tool_args, str):
                try:
                    import json

                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}

            try:
                result = _resolve_callable(tool_fn)(**(tool_args or {}))
                content = str(result)
            except Exception as e:
                content = f"Tool '{tool_name}' execution failed: {e}"

            messages.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_name": tool_name,
                }
            )

    # Safety exit: do not raise, just return the last response.
    return last_response, messages


def add(a: float, b: float) -> float:
    """Apply addition"""
    return a + b


def sub(a: float, b: float) -> float:
    """Apply subtraction"""
    return a - b


def mul(a: float, b: float) -> float:
    """Apply multiplication"""
    return a * b


def div(a: float, b: float) -> float:
    """Apply division"""
    return a / b


if __name__ == "__main__":
    from ollama import chat

    response, messages = ollama_automatic_function_calling(
        chat,
        model="glm-5:cloud",
        tools=[add, sub, mul, div],
        messages=[{"role": "user", "content": "这个数学题：4+7*9/6是多少?"}],
        think="low",  # or true, false
        options={
            "temperature": 1,
            "top_p": 1.0,
            "top_k": 60,
        },
    )
