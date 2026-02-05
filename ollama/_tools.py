from functools import wraps

_list_tools = []
_async_list_tools = []

def ollama_async_tool(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    _async_list_tools.append(wrapper)
    return wrapper

def ollama_tool(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    _list_tools.append(wrapper)
    return wrapper

def get_ollama_tools_name():
    list_name_tools = {} 
    for func in _list_tools + _async_list_tools:
        if func.__name__ not in list_name_tools:
            list_name_tools[func.__name__] = func
    return list_name_tools

def get_ollama_tools():
    return _list_tools + _async_list_tools

def get_ollama_name_async_tools():
    return {f"{func.__name__}" for func in _async_list_tools}

def get_ollama_tool_description():
    from ollama._utils import _parse_docstring
    result = {}
    for func in _list_tools + _async_list_tools:
        if func.__doc__:
            parsed_docstring = _parse_docstring(func.__doc__)
            if parsed_docstring and str(hash(func.__doc__)) in parsed_docstring:
                result[func.__name__] = parsed_docstring[str(hash(func.__doc__))].strip()
    
    return result
