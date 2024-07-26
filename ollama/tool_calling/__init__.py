import json
import typing as t

from .._types import Tool
from ._utils import (
    PydanticModel, TDict, NTuple, ParseError, generate_tool_schema, invoke_tool
)

__all__ = [
    'ToolRegistry',
    'ToolRegisteredError',
    'ToolNotRegisteredError',
    'ParseError'
]

P = t.ParamSpec("P")
R = t.TypeVar("R")

class ToolRegisteredError(Exception):
    ...

class ToolNotRegisteredError(Exception):
    ...

class ToolRegistry:
    """
    A versatile registry designed to manage a variety of tools, including 
    `(a)sync` function, `pydantic.BaseModel`, `typing.TypedDict`, and `typing.NamedTuple`. 
    
    It provides the capability to generate schemas for these tools, which are essential for LLM tool-calling. 

    Additionally, it allows for the invocation of tools using their metadata -- name & raw arguments.
    """

    def __init__(self, *, override: bool = False) -> None:
        """
        Create a new tool registry.

        :param override: When set to True, allows the new tool to replace a previously registered tool with the same name.
        """
        self._override = override
        self._tool_map = {}
    
    @t.overload
    def register(self, __obj: t.Type[PydanticModel]) -> t.Type[PydanticModel]:
        """
        Register a pydantic model by decorating it or passing it as an argument.

        :param __obj: Pydantic model to register.

        Example:
        >>> from pydantic import BaseModel
        >>> from typing import Literal
        >>> 
        >>> @tool_registry.register
        ... class User(BaseModel):
        ...     '''User Information'''
        ...     name: str
        ...     role: Literal['admin', 'developer', 'tester']
        """
    @t.overload
    def register(self, __obj: t.Type[TDict]) -> t.Type[TDict]:
        """
        Register a typed dict by decorating it or passing it as an argument.

        :param __obj: TypedDict model to register.

        Example:
        >>> from typing import TypedDict, Literal
        >>> 
        >>> @tool_registry.register
        ... class User(TypedDict):
        ...     '''User Information'''
        ...     name: str
        ...     role: Literal['admin', 'developer', 'tester']
        """
    @t.overload
    def register(self, __obj: t.Type[NTuple]) -> t.Type[NTuple]:
        """
        Register a named tuple by decorating it or passing it as an argument.

        :param __obj: NamedTuple model to register.

        Example:
        >>> from typing import NamedTuple, Literal
        >>> 
        >>> @tool_registry.register
        ... class User(NamedTuple):
        ...     '''User Information'''
        ...     name: str
        ...     role: Literal['admin', 'developer', 'tester']
        """
    @t.overload
    def register(self, __obj: t.Callable[P, t.Awaitable[R]]) -> t.Callable[P, t.Awaitable[R]]:
        """
        Register an async function by decorating it or passing it as an argument.

        :param __obj: Async function to register.

        Example:
        >>> @tool_registry.register
        ... async def reverse_string(string: str):
        ...     '''Reverse the given string'''
        ...     return string[::-1]
        """
    @t.overload
    def register(self, __obj: t.Callable[P, R]) -> t.Callable[P, R]:
        """
        Register a function by decorating it or passing it as an argument.

        :param __obj: Function to register.

        Example:
        >>> @tool_registry.register
        ... def reverse_string(string: str):
        ...     '''Reverse the given string'''
        ...     return string[::-1]
        """
    def register(
        self, 
        __obj: t.Callable[P, R] | t.Callable[P, t.Awaitable[R]] | t.Type[PydanticModel | TDict | NTuple]
    ):
        name = __obj.__name__
        reg_obj = self._tool_map.get(name)
        if reg_obj and reg_obj is not __obj and not self._override:
            raise ToolRegisteredError(
                f"Tool with name {reg_obj.__name__!r} is already registered."
            )
            
        self._tool_map[name] = __obj
        return __obj
    
    def register_multiple(
        self, 
        *__objs: t.Callable[P, R] | t.Callable[P, t.Awaitable[R]] | t.Type[PydanticModel | TDict | NTuple]
    ):
        """
        Register multiple objects at once.

        :param __objs: Objects to register.

        Example:
        >>> def reverse_string(string: str):
        ...     '''Reverse the given string'''
        ...     return string[::-1]
        >>> 
        >>> class User(BaseModel):
        ...     '''User Information'''
        ...     name: str
        ...     role: Literal['admin', 'developer', 'tester']
        >>> 
        >>> tool_registry.register_multiple(reverse_string, User)
        """
        for obj in __objs:self.register(obj)

    @property
    def tools(self) -> list[Tool]:
        """Schema representation of registered tools."""
        return [generate_tool_schema(obj) for obj in self._tool_map.values()]
    
    def invoke(self, name: str, arguments: str | t.Dict[str, t.Any]):
        if (obj := self._tool_map.get(name)) is None:
            raise ToolNotRegisteredError(f"{name!r} tool has not been registered")
        
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                raise ParseError("arguments is not a valid JsON object")
            
        return invoke_tool(obj, arguments=arguments)