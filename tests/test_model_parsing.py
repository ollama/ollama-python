import pytest
from ollama._types import Message

def test_damaged_cloud_output_parsing():
    """
    Test that raw text containing text-embedded tool-call structures 
    is intercepted and parsed correctly into Message objects.
    """

    simulated_server_payload = {
        "role": "assistant",
        "content": '{"params":{"query": "AI research"},"task":"Describe document"}<|call|>commentary'
    }
    
    # This invokes your custom handle_embedded_cloud_tool_calls model validator!
    message_obj = Message(**simulated_server_payload)
    assert message_obj.content == "commentary"
    assert message_obj.tool_calls is not None
    assert len(message_obj.tool_calls) == 1
    extracted_tool = message_obj.tool_calls[0]
    assert extracted_tool.function.name == "Describe document"
    assert extracted_tool.function.arguments == {"query": "AI research"}
