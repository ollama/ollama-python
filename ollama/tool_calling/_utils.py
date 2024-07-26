import inspect
import asyncio
import typing as t
from enum import Enum
from pathlib import Path
from pydantic import BaseModel
from docstring_parser import parse_from_object, Docstring

_SUPPORTED_TYPE_MAP = {
    # Builtins
    str: 'string',
    int: 'integer',
    float: 'integer',
    bool: 'boolean',
    list: 'array',
    dict: 'object',
     
    Path: 'string',
    t.List: 'array',
    t.Dict: 'object',

    # object with properties
    t.NamedTuple: 'object',
    t.TypedDict: 'object',
    BaseModel: 'object',

    # enums
    t.Literal: 'string',
    Enum: 'string', 
}
"""Supported types mapped property types"""

_SUPPORTED_TYPES_REPR = " | ".join(
    f"`{t.__module__.split('.')[0]}.{t.__name__}`" 
    if t.__module__ != 'builtins' else f"`{t.__name__}`" for t in _SUPPORTED_TYPE_MAP
)

class ParseError(Exception):
    ...

class ParamMetadata(t.NamedTuple):
    label: str
    schema: t.Dict[str, t.Any]
    required: bool

def create_parameters_schema(__params: t.Iterable[ParamMetadata]) -> t.Dict[str, t.Any]:
    properties, required_props = {}, []
    for (label, schema, required) in __params:
        properties[label] = schema
        if required:
            required_props.append(label)
    
    if not properties:
        return {}
    
    return {
        "type": "object", "properties": properties, "required": required_props
    }

def map_param_to_description(docstring: Docstring) -> t.Dict[str, str]:
    description_map = {}
    for param in docstring.params:
        if param.description:
            description_map[param.arg_name] = param.description
    return description_map

def has_orig_bases(__obj, __base: str):
    if (
        '__orig_bases__' in __obj.__dict__ and \
        __obj.__orig_bases__[0].__name__ == __base
    ):
        return True
    return False

# Function-related definitions

P = t.ParamSpec("P")
R = t.TypeVar("R")

def is_async(__fn: t.Callable[..., t.Any]) -> bool:
    """Returns true if the callable is async, accounting for wrapped callables"""
    is_coroutine = inspect.iscoroutinefunction(__fn)
    while hasattr(__fn, "__wrapped__"):
        __fn = __fn.__wrapped__  # type: ignore - dynamic
        is_coroutine = is_coroutine or inspect.iscoroutinefunction(__fn)
    return is_coroutine

def run_async(coro: t.Awaitable[R]) -> R:
    """
    Gets an existing event loop to run the coroutine. 
    If there is no existing event loop, creates a new one.
    """
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError as e:
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            raise RuntimeError(
                "Detected nested async. Please use nest_asyncio.apply() to allow nested event loops."
            )

def generate_function_metadata(
    __fn: t.Callable[..., t.Any], description_map: t.Dict[str, str]
):
    for label, param in inspect.signature(__fn).parameters.items():
        schema = parse_property(param.annotation)
        if label in description_map:
            schema['description'] = description_map[label]

        yield ParamMetadata(
            label=label,
            schema=schema,
            required=True if param.default is inspect._empty else False
        )

nest_asyncio_err = "cannot be called from a running event loop"
nest_asyncio_msg = "The event loop is already running. Add `import nest_asyncio; nest_asyncio.apply()` to your code to fix this issue."

def compile_function_object(
    __fn: t.Callable[P, R] | t.Callable[P, t.Awaitable[R]], 
    arguments: t.Dict[str, t.Any]
) -> R:
    args, kwargs = [], {}
    for key, param in inspect.signature(__fn).parameters.items():
        default, raw_value = param.default, None
        raw_value = arguments.get(key)
        if raw_value is None and default is inspect._empty:
            raise ParseError(f"{key!r} parameter is required.")
        
        value = parse_value(param.annotation, default if raw_value is None else raw_value)
        if param.kind is inspect._ParameterKind.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[key] = value

    if not is_async(__fn):
        return __fn(*args, **kwargs)
    
    try:
        return run_async(__fn(*args, **kwargs)) 
    except RuntimeError as e:
        if nest_asyncio_err in str(e):
            raise RuntimeError(nest_asyncio_msg)
        else:
            raise e
        

# Pydantic-related definitions
PydanticModel = t.TypeVar("PydanticModel", bound=BaseModel)

def is_pydantic_model(__obj):
    is_class = isinstance(__obj, type)
    import sys

    if sys.version_info < (3, 10):
        if len(typing.get_args(__obj)) == 0:
            return False
        return is_class and issubclass(typing.get_args(__obj)[0], BaseModel)
    return is_class and issubclass(__obj, BaseModel)

def generate_pydantic_metadata(
    __model: t.Type[PydanticModel], description_map: t.Dict[str, str]
):
    for label, field in __model.model_fields.items():
        if field.annotation is __model:
            raise ParseError(
                f"{label!r} field cannot have the same type as the Pydantic model {__model.__name__!r}."
            )
        schema = parse_property(field.annotation)
        if description := field.description or description_map.get(label):
            schema['description'] = description

        yield ParamMetadata(
            label=label,
            schema=schema,
            required=True if field.is_required() else False
        )

def compile_pydantic_object(
    __model: t.Type[PydanticModel], arguments: t.Dict[str, t.Any]
):
    name, fields = __model.__name__, {}
    for key, field in __model.model_fields.items():
        if key not in arguments and field.is_required():
            raise ParseError(
                f"{name!r} model required field {key!r} missing."
            )
        fields[key] = parse_value(field.annotation, arguments.get(key, field.default))
    return __model(**fields)


# TypedDict-related definitions
class TypedDictProtocol(t.Protocol):
    __annotations__: t.Dict[str, t.Type]
    __required_keys__: t.FrozenSet[str]
    __optional_keys__: t.FrozenSet[str]

TDict = t.TypeVar("TDict", bound=TypedDictProtocol)

def is_typeddict(__obj):
    return has_orig_bases(__obj, "TypedDict")

def generate_typeddict_metadata(
    __td: t.Type[TDict], description_map: t.Dict[str, str]
):
    for label, annotation in t.get_type_hints(__td).items():
        if is_typeddict(annotation) and annotation.__name__ == __td.__name__:
            raise ParseError(
                f"{label!r} field cannot have the same type as the TypeDict class {__td.__name__!r}."
            )
        schema = parse_property(annotation)
        if label in description_map:
            schema['description'] = description_map[label]

        yield ParamMetadata(
            label=label, 
            schema=schema, 
            required=False if label in __td.__dict__ else True
        )

def compile_typeddict_object(
    __td: t.Type[TDict], arguments: t.Dict[str, t.Any]
):
    name, fields = __td.__name__, {}
    for key, annotation in t.get_type_hints(__td).items():
        if key not in arguments and key not in __td.__dict__:
            raise ParseError(
                f"{name!r} TypedDict required field {key!r} missing."
            )
        fields[key] = parse_value(annotation, arguments.get(key, __td.__dict__.get(key)))
    return __td(**fields)


# NamedTuple-related definitions
NTFields = t.TypeVar("NTFields", bound=tuple[str, ...])

class NamedTupleProtocol(t.Protocol[NTFields]):
    _fields: NTFields
    _field_defaults: t.Dict[str, t.Any]
    __annotations__: t.Dict[str, t.Type]

NTuple = t.TypeVar("NTuple", bound=NamedTupleProtocol)

def is_namedtuple(__obj):
    return has_orig_bases(__obj, "NamedTuple")

def generate_namedtuple_metadata(
    __nt: t.Type[NTuple], description_map: t.Dict[str, str]
):
    for label, annotation in t.get_type_hints(__nt).items():
        if is_namedtuple(annotation) and annotation.__name__ == __nt.__name__:
            raise ParseError(
                f"{label!r} field cannot have the same type as the NamedTuple class {__nt.__name__!r}."
            )
        schema = parse_property(annotation)
        if label in description_map:
            schema['description'] = description_map[label]

        yield ParamMetadata(
            label=label, 
            schema=schema, 
            required=False if label in __nt._field_defaults else True
        )

def compile_namedtuple_object(
    __nt: t.Type[NTuple], arguments: t.Dict[str, t.Any]
):
    name, fields = __nt.__name__, {}
    for key, annotation in t.get_type_hints(__nt).items():
        if key not in arguments:
            raise ParseError(
                f"{name!r} NamedTuple required field {key!r} missing."
            )
        fields[key] = parse_value(annotation, arguments.get(key))
    return __nt(**fields)

#---
def parse_property(__annotation: t.Type | t.ForwardRef) -> t.Dict[str, t.Any]:
    """
    Parse the annotation to tool-calling specific property map
    """
    if isinstance(__annotation, t.ForwardRef):
        __annotation = __annotation._evaluate({}, {}, frozenset())

    origin: t.Type = t.get_origin(__annotation) or __annotation
    args = t.get_args(__annotation)
    if args:
        if origin is t.Literal:
            return {'type': 'string', 'enum': list(args)}
        if origin in (list, t.List):
            return {'type': 'array', 'items': parse_property(args[0])}
        
    if issubclass(origin, Path):
        return {'type': 'string'}
        
    if issubclass(origin, Enum):
        return {'type': 'string', 'enum': origin._member_names_}

    if (_type := _SUPPORTED_TYPE_MAP.get(origin)) is not None:
        return {'type': _type}
    
    generate_fn = None
    if is_pydantic_model(origin):
        generate_fn = generate_pydantic_metadata
    elif is_typeddict(origin):
        generate_fn = generate_typeddict_metadata
    elif is_namedtuple(origin):
        generate_fn = generate_namedtuple_metadata
    
    if generate_fn is not None:
        return create_parameters_schema(
            generate_fn(
                origin, 
                map_param_to_description(parse_from_object(origin))
            )
        )
    raise ParseError(f"{origin.__name__!r} type is not supported.\nSupported types: {_SUPPORTED_TYPES_REPR}")

def parse_value(__annotation: t.Type | t.ForwardRef, raw_value: t.Any):
    """
    Parse the raw value as instance of the given annotation.
    """
    rest_err = lambda v: f"but received value of type {type(v)!r} instead."
    if isinstance(__annotation, t.ForwardRef):
        __annotation = __annotation._evaluate({}, {}, frozenset())
    
    origin: t.Type = t.get_origin(__annotation) or __annotation
    args = t.get_args(__annotation)
    if args:
        if origin is t.Literal:
            if not isinstance(raw_value, str):
                raise ParseError(
                    f"Expected string value for Literal parameter, {rest_err(raw_value)}" 
                )
            if raw_value not in args:
                raise ParseError(
                    f"{raw_value!r} is not a valid literal. Valid literals: {args!r}"
                )
            return raw_value
            
        if origin in (list, t.List):
            if not isinstance(raw_value, list):
                raise ParseError(f"Expected list value, {rest_err(raw_value)}")
            return [parse_value(args[0], e) for e in raw_value]
                
    if issubclass(origin, Path):
        if not isinstance(raw_value, str):
            raise ParseError(
                f"Expected string value for `pathlib.Path` type, {rest_err(raw_value)}"
            )
        return Path(raw_value)
    
    if issubclass(origin, Enum):
        if not isinstance(raw_value, str):
            raise ParseError(
                f"Expected string value for {origin.__name__}, {rest_err(raw_value)}"
            )
        if (enum := origin._member_map_.get(raw_value)) is None:
            raise ParseError(
                f"{raw_value!r} is not a valid {origin.__name__!r} member." 
                f"Valid members: {origin._member_names_!r}"
            )
        return enum
    
    if origin in _SUPPORTED_TYPE_MAP:
        if not isinstance(raw_value, origin):
            raise ParseError(
                f"Expected parameter type {origin.__name__!r}, {rest_err(raw_value)}" 
            )
        return raw_value
    
    compile_fn = None
    if is_pydantic_model(origin):
        compile_fn = compile_pydantic_object            
    elif is_typeddict(origin):
        compile_fn = compile_typeddict_object
    elif is_namedtuple(origin):
        compile_fn = compile_namedtuple_object

    if compile_fn is not None:  
        if not isinstance(raw_value, dict):
            raise ParseError(
                f"Expected dictionary value for {origin.__name__}, {rest_err(raw_value)}"
            )
        return compile_fn(origin, raw_value)
    
    raise ParseError(
        f"{origin.__name__!r} type is not supported.\nSupported types: {_SUPPORTED_TYPES_REPR}"
    )

def generate_tool_schema(__obj):
    if is_pydantic_model(__obj):
        generate_fn = generate_pydantic_metadata
    elif is_typeddict(__obj):
        generate_fn = generate_typeddict_metadata
    elif is_namedtuple(__obj):
        generate_fn = generate_namedtuple_metadata
    elif callable(__obj):
        generate_fn = generate_function_metadata
    else:
        ParseError(f"Schema generation failed, given object is not supported")
    
    docstring = parse_from_object(__obj)
    parameters = create_parameters_schema(
        generate_fn(
            __obj, map_param_to_description(docstring)
        )
    )
    fn_schema = {'name': __obj.__name__, 'parameters': parameters}
    
    if (desc := docstring.short_description):
        fn_schema['description'] = desc

    return {'type': 'function', 'function': fn_schema}

def invoke_tool(__obj: t.Any, *, arguments: t.Dict[str, t.Any]):
    if is_pydantic_model(__obj):
        compile_fn = compile_pydantic_object            
    elif is_typeddict(__obj):
        compile_fn = compile_typeddict_object
    elif is_namedtuple(__obj):
        compile_fn = compile_namedtuple_object
    elif callable(__obj):
        compile_fn = compile_function_object
    else:
        ParseError(f"Tool invocation failed, given object is not supported")
    
    return compile_fn(__obj, arguments=arguments)