import inspect
from enum import EnumMeta

try:
    # Python 3.10+
    from types import NoneType
except ImportError:
    NoneType = type(None)

from typing import (
    Annotated,
    Dict,
    List,
    Sequence,
    TypedDict,
    Union,
    get_origin as get_type_origin,
    get_args as get_type_args,
)
from ._types import Tool, ToolFunction, Parameters, Property


class CannotParseParameters(Exception):
    pass


PYTHON_TYPE_TO_SCHEMA_TYPE = {
    bool: "boolean",
    float: "number",
    int: "integer",
    list: "array",
    str: "string",
}


class ParsedAnnotation(TypedDict):
    description: str
    is_optional: bool
    param_type: str
    property_enum: Sequence[str]  # matches Property.enum


def _get_param_info(param: inspect.Parameter) -> ParsedAnnotation:
    """Given an inspected function's Parameter, determine
    the relevant ToolFunction parameter/property metadata"""
    is_optional = param.default not in (inspect.Parameter.empty, NoneType)
    atn, raw_type = param.annotation, None
    enum_values: List[str] = None
    if get_type_origin(atn) == Annotated:
        # we expect the outer most Annotation is the one that
        # contains the description text
        desc_text = atn.__metadata__[0]
    else:
        # fallback strategy, just wordify the snake case param name
        desc_text = param.name.replace("_", " ")

    if get_type_origin(atn) is None:
        # support/gracefully handle un-Annotated params
        # e.g. param type just one of the built-in python types
        if isinstance(atn, EnumMeta):
            # support unannotated e.g. 'foo: ExampleEnum = ExampleEnum.THING'
            raw_type = str
            enum_values: List[str] = [e.value for e in atn]
        else:
            # e.g. the type hint is just a basic type e.g. str, int, float
            raw_type = atn
        return ParsedAnnotation(
            param_type=raw_type,
            is_optional=is_optional,
            description=desc_text,
            property_enum=enum_values,
        )

    cur_atn = atn
    while raw_type is None:
        # we need to keep drilling into nested Annotations to find
        # the user's intended function param type
        try:
            # e.g. Annotated[str, "description"] has two args
            #   index 0: the 'str' class
            #   index 1: the text description
            if get_type_origin(cur_atn) is list:
                # found Annotated[List[...]]
                raw_type = list
            elif isinstance(cur_atn, EnumMeta):
                raw_type = str  # see Property.enum support scope
                enum_values: List[str] = [e.value for e in cur_atn]
            else:
                candidate = get_type_args(cur_atn)[0]
                if candidate in PYTHON_TYPE_TO_SCHEMA_TYPE:
                    raw_type = candidate
        except IndexError:
            raise CannotParseParameters(
                f"ERROR: not expected to parse {param.name=} "
                f"where {param.annotation=}"
            )

        if get_type_origin(cur_atn) == Union and get_type_args(cur_atn)[1] is NoneType:
            # found Optional annotation, i.e. Optional is actually Union[Any, None]
            is_optional = True

        # drill down to the next annotation level
        cur_atn = candidate

    return ParsedAnnotation(
        param_type=raw_type,
        is_optional=is_optional,
        description=desc_text,
        property_enum=enum_values,
    )


PYTHON_TYPE_TO_SCHEMA_TYPE = {
    bool: "boolean",
    float: "number",
    int: "integer",
    list: "array",
    str: "string",
}


def annotated_tool(orig_fn) -> Tool:
    """
    Decorates a typing-annotated function and reflects on its parameter properties to automatically
    create the necessary ToolFunction + Parameter + Property schema for the tools= field in ollama.chat
    """
    sig = inspect.signature(orig_fn)
    props: Dict[str, Property] = {}
    required_params = set(sig.parameters.keys())
    for param_name, param in sig.parameters.items():
        if not param.annotation:
            # warn this parameter will be ignored by the ollama ToolFunction spec
            continue
        param_info = _get_param_info(param)

        if param_info["is_optional"]:
            required_params.discard(param_name)

        if param_info["param_type"] is None:
            raise Exception(param_name)

        optional_prop_kwargs = {}
        if "property_enum" in param_info:
            # don't put enum in the Property if not present in annotation
            optional_prop_kwargs["enum"] = param_info["property_enum"]

        props[param_name] = Property(
            type=PYTHON_TYPE_TO_SCHEMA_TYPE[param_info["param_type"]],
            description=param_info["description"],
            is_optional=param_info["is_optional"],
            **optional_prop_kwargs,
        )
    orig_fn.tool_schema = {
        "type": "function",
        "function": ToolFunction(
            name=orig_fn.__name__,
            description=orig_fn.__doc__,
            parameters=Parameters(
                type="object", required=list(required_params), properties=props
            ),
        ),
    }
    return orig_fn
