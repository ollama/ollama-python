def test_tool_and_async_tool_registration():
    import types
    from ollama import _tools
    _tools._list_tools.clear()
    _tools._async_list_tools.clear()

    @(_tools.ollama_tool)
    def t1():
        return "ok"

    @(_tools.ollama_async_tool)
    async def t2():
        return "ok"

    assert t1 in _tools._list_tools
    assert t2 in _tools._async_list_tools
    assert t1() == "ok"
    import asyncio
    assert asyncio.run(t2()) == "ok"

def test_get_tools_name_and_get_tools():
    from ollama import _tools
    _tools._list_tools.clear()
    _tools._async_list_tools.clear()

    @(_tools.ollama_tool)
    def t3():
        return 1
    @(_tools.ollama_async_tool)
    async def t4():
        return 2

    names = _tools.get_ollama_tools_name()
    assert "t3" in names
    assert "t4" in names
    assert callable(names["t3"])
    assert callable(names["t4"])
    tools = _tools.get_ollama_tools()
    assert t3 in tools
    assert t4 in tools

def test_get_ollama_name_async_tools():
    from ollama import _tools
    _tools._list_tools.clear()
    _tools._async_list_tools.clear()

    @(_tools.ollama_tool)
    def sync_tool():
        return 1

    @(_tools.ollama_async_tool)
    async def async_tool1():
        return 2

    @(_tools.ollama_async_tool)
    async def async_tool2():
        return 3

    async_names = _tools.get_ollama_name_async_tools()

    assert "async_tool1" in async_names
    assert "async_tool2" in async_names
    assert "sync_tool" not in async_names
    assert len(async_names) == 2

def test_get_ollama_tool_description():
    from ollama import _tools
    _tools._list_tools.clear()
    _tools._async_list_tools.clear()

    @(_tools.ollama_tool)
    def tool_with_doc():
        """
        Test description for sync tool.
        """
        return 1

    @(_tools.ollama_async_tool)
    async def async_tool_with_doc():
        """
        Test description for async tool.
        """
        return 2

    @(_tools.ollama_tool)
    def tool_without_doc():
        return 3

    descriptions = _tools.get_ollama_tool_description()
    
    
    assert "tool_with_doc" in descriptions
    assert "async_tool_with_doc" in descriptions
    assert "tool_without_doc" not in descriptions  
    
    assert "Test description for sync tool" in descriptions["tool_with_doc"]
    assert "Test description for async tool" in descriptions["async_tool_with_doc"]
