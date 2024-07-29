from enum import Enum
import typing as t
import ollama._toolfuncs as ollama_tools


class ExampleEnum(Enum):
    FOO = "foo"
    BAR = "bar"
    BAZ = "baz"


@ollama_tools.annotated_tool
def example_1(
    expr_1: t.Annotated[str, "first simple str value"],
    expr_2: t.Annotated[str, "second simple str value"],
    req_int_arg: t.Annotated[int, "a required integer"],
    req_float_arg: t.Annotated[float, "any real number"],
    req_list_arg_1: t.Annotated[list, "any builtin list"],
    req_list_arg_2: t.Annotated[t.List[t.Any], "any typed List"],
    req_enum_1: t.Annotated[ExampleEnum, "required foo bar or baz"],
    opt_arg_1: t.Annotated[t.Optional[str], "an optional string"] = None,
    opt_enum_1: t.Annotated[t.Optional[ExampleEnum], "optional foo bar or baz"] = None,
    opt_enum_2: ExampleEnum = ExampleEnum.FOO,
    opt_builtin_str_1: str = "foobar",
    opt_builtin_int_1: int = 1e6,
    opt_builtin_float_1: float = 1.0,
    opt_builtin_list_1: list = None,
):
    """a test case for Annotating a tool function's parameters.
    the docstring of the function is the Function.description."""
    return NotImplementedError("not in test scope to check function response")


def test_function_description():
    assert example_1.tool_schema["function"]["description"] == example_1.__doc__


def test_annotated_str():
    for k in ["expr_1", "expr_2"]:
        _ = example_1.tool_schema["function"]["parameters"]["properties"][k]
        assert _["type"] == "string"
        assert "simple str value" in _["description"]
        assert _["is_optional"] is False


def test_annotated_int():
    _ = example_1.tool_schema["function"]["parameters"]["properties"]["req_int_arg"]
    assert _["type"] == "integer"
    assert _["description"] == "a required integer"
    assert _["is_optional"] is False


def test_annotated_float():
    _ = example_1.tool_schema["function"]["parameters"]["properties"]["req_float_arg"]
    assert _["type"] == "number"
    assert _["description"] == "any real number"
    assert _["is_optional"] is False


def test_annotated_builtin_list():
    _ = example_1.tool_schema["function"]["parameters"]["properties"]["req_list_arg_1"]
    assert _["type"] == "array"
    assert _["description"] == "any builtin list"
    assert _["is_optional"] is False


def test_annotated_typed_list():
    _ = example_1.tool_schema["function"]["parameters"]["properties"]["req_list_arg_2"]
    assert _["type"] == "array"
    assert _["description"] == "any typed List"
    assert _["is_optional"] is False


def test_annotated_optional_builtin_str():
    _ = example_1.tool_schema["function"]["parameters"]["properties"][
        "opt_builtin_str_1"
    ]
    assert _["type"] == "string"
    assert _["description"] == "opt builtin str 1"
    assert _["is_optional"] is True


def test_annotated_optional_builtin_int():
    _ = example_1.tool_schema["function"]["parameters"]["properties"][
        "opt_builtin_int_1"
    ]
    assert _["type"] == "integer"
    assert _["description"] == "opt builtin int 1"
    assert _["is_optional"] is True


def test_annotated_optional_builtin_float():
    _ = example_1.tool_schema["function"]["parameters"]["properties"][
        "opt_builtin_float_1"
    ]
    assert _["type"] == "number"
    assert _["description"] == "opt builtin float 1"
    assert _["is_optional"] is True


def test_annotated_optional_builtin_list():
    _ = example_1.tool_schema["function"]["parameters"]["properties"][
        "opt_builtin_list_1"
    ]
    assert _["type"] == "array"
    assert _["description"] == "opt builtin list 1"
    assert _["is_optional"] is True


def test_annotated_enum_1():
    _ = example_1.tool_schema["function"]["parameters"]["properties"]["req_enum_1"]
    assert _["type"] == "string"
    assert _["description"] == "required foo bar or baz"
    assert _["is_optional"] is False


def test_optional_annotated_enum_1():
    _ = example_1.tool_schema["function"]["parameters"]["properties"]["opt_enum_1"]
    assert _["type"] == "string"
    assert _["description"] == "optional foo bar or baz"
    assert _["is_optional"] is True


def test_optional_unannotated_enum_1():
    _ = example_1.tool_schema["function"]["parameters"]["properties"]["opt_enum_2"]
    assert _["type"] == "string"
    assert _["description"] == "opt enum 2"
    assert _["is_optional"] is True
