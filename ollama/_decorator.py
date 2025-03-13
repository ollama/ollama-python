import inspect
import functools
import importlib.util
import re

# Global list to store schema information + references to actual Python functions.
ollama_tools = []

TYPE_MAPPING = {
    "int": "integer",
    "float": "number",
    "str": "string",
    "bool": "boolean",
    "list": "array",
    "tuple": "array",
    "dict": "object",
    "None": "null",
}

def get_json_type(py_type_str: str) -> str:
    return TYPE_MAPPING.get(py_type_str, "string")

def parse_param_docstring(docstring: str) -> dict:
    """
    A simple parser for lines like:

        Args:
          param_name (type): description

    Adjust the regex/logic to match your docstring style.
    """
    param_desc = {}
    if not docstring:
        return param_desc

    docstring = inspect.cleandoc(docstring)
    pattern = re.compile(r'^\s*(\w+)\s*\(([^)]+)\):\s*(.*)$', re.MULTILINE)
    matches = pattern.findall(docstring)
    for match in matches:
        param_name, _, description = match
        param_desc[param_name] = description
    return param_desc

def create_param_schema(param, param_doc_desc=None):
    annotation_str = str(param.annotation)
    base_type = annotation_str.split("[", 1)[0].replace("<class '", "").replace("'>", "")
    schema_type = get_json_type(base_type.split(".")[-1])

    param_schema = {"type": schema_type, "description": param_doc_desc or ""}

    if schema_type == "array" and "[" in annotation_str and "]" in annotation_str:
        item_type_str = annotation_str.split("[", 1)[1].rstrip("]")
        item_base_type = item_type_str.replace("<class '", "").replace("'>", "").split(".")[-1]
        param_schema["items"] = {"type": get_json_type(item_base_type)}

    return param_schema

def ollamafunc(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    pydantic_found = importlib.util.find_spec("pydantic") is not None
    if pydantic_found:
        from pydantic import BaseModel

    signature = inspect.signature(func)
    docstring = inspect.cleandoc(func.__doc__ or "")
    param_doc = parse_param_docstring(docstring)

    properties = {}
    required = []

    for param_name, param in signature.parameters.items():
        if pydantic_found and issubclass(param.annotation, BaseModel):
            # Handle Pydantic model as object
            model_schema = param.annotation.schema()
            properties[param_name] = {
                "type": "object",
                "description": model_schema.get("description", ""),
                "properties": {},
                "required": model_schema.get("required", [])
            }
            for field_name, field_props in model_schema.get("properties", {}).items():
                properties[param_name]["properties"][field_name] = {
                    "type": field_props.get("type", "string"),
                    "description": field_props.get("description", "")
                }
            required.append(param_name)
        else:
            # Basic type
            desc = param_doc.get(param_name, "")
            properties[param_name] = create_param_schema(param, desc)
            required.append(param_name)

    tool_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": docstring,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

    ollama_tools.append({"schema": tool_schema, "reference": func})
    return wrapper

def get_ollama_tools():
    return ollama_tools
