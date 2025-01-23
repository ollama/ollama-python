import asyncio
from ollama import ChatResponse, AsyncClient
import ollama
from pydantic import BaseModel, ValidationError, validator
from typing import Any, List, Optional, Type, Literal, Dict, Union
from json.decoder import JSONDecodeError
import json
from pathlib import Path
from datetime import datetime
import logging

########################################################
# Constants and Configuration
########################################################
# Ollama (https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)
# 3.2 model -> "3.2:3B+ supports function calling, but llama3.2:1b requires a tool call description added to the system prompt

MODEL = 'llama3.2:1b'
# MODEL = 'llama3.2:3b'
MAX_TOOL_RETRIES = 2
conversation_history = []
truncate_history = 5 # Last 5 messages

BASE_DIR = Path('data')
CONVERSATION_DIR = BASE_DIR / 'conversations'
# Ensure directories exist for logging
for dir_path in [CONVERSATION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Agent Names
AGENT_NAMES = ['Addition Agent', 'Subtraction Agent', 'Multiplication Agent', 'Division Agent']

########################################################
# Agent Delegation Data Schema
########################################################
class AgentDelegation(BaseModel):
    agent_name: str
    query: str
    reason: str
    
    # check if agent name is valid (AGENT_NAMES)
    @validator('agent_name')
    def validate_agent_name(cls, v):
        if v not in AGENT_NAMES:
            raise ValueError(f'Agent name must be one of {AGENT_NAMES}')
        return v
    
class AgentDelegationResponse(BaseModel):
    delegation: AgentDelegation
    response: str
    
########################################################
# Operator Agent
########################################################
class OperatorAgent:
    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = ollama.AsyncClient()
        
        # Define tools map as part of agent initialization  
        self.tools_map = {
            'delegate_to_agent': self.delegate_to_agent,
        }
        
        self.messages = []
        self.available_tools = [
            {
                "type": "function",
                "function": {
                    "name": "delegate_to_agent",
                    "description": "Delegate the query to the appropriate agent.",
                    "parameters": {
                        "type": "object",
                        "required": ["agent_name", "query", "reason"],
                        "properties": {
                            "agent_name": {"type": "string", "enum": AGENT_NAMES},
                            "query": {"type": "string"},
                            "reason": {"type": "string"}
                        }
                    }
                }
            },
        ]
        
        # Validation Schema Map (Pydantic Model for Each Tool)
        self.validation_schema_map = {
            'delegate_to_agent': AgentDelegation,
        }

        # Add output schema property
        self.output_schema = AgentDelegationResponse
        
        # System Prompt (Dynamic Tool Descriptions)
        self.system_prompt = f'''
            <system>
                You are an expert in delegating queries to the appropriate agent.
                
                <tools>
                {self.available_tools}
                </tools>
                
                <rules>
                1. ONLY call a tool when you have all required parameters
                2. If agent not available, respond: "Sorry, I can only delegate to the Addition Agent, Subtraction Agent, Multiplication Agent, or Division Agent."
                3. If parameters missing, ask for them
                4. For tool responses, extract ONLY the number from "result=X" for the final output
                </rules>
                
                <format>
                    <input>what is 5 plus 3?</input>
                    <tool_call>delegate_to_agent(agent_name=<agent_name>, query=<query>, reason=<reason>)</tool_call>
                </format>
                
                <instructions>
                DO NOT explain the delegation process.
                DO NOT make new tool calls after receiving a response.
                ONLY use the delegate_to_agent function for ALL operations.
                NEVER try to call agents directly.
                
                Addition Agent:
                - Handles addition queries
                
                Subtraction Agent:
                - Handles subtraction queries
                
                Multiplication Agent:
                - Handles multiplication queries
                
                Division Agent:
                - Handles division queries
                
                </instructions>
                
                <example>
                User: what is 5 plus 3?
                Assistant: delegate_to_agent(agent_name='Addition Agent', query='5 + 3', reason='query involves addition')
                </example>
                
                <example>
                User: what is 5 minus 3?
                Assistant: delegate_to_agent(agent_name='Subtraction Agent', query='5 - 3', reason='query involves subtraction')
                </example>
                
                <example>
                User: what is 5 times 3?
                Assistant: delegate_to_agent(agent_name='Multiplication Agent', query='5 * 3', reason='query involves multiplication')
                </example>
                
                <example>
                User: what is 72 divided by 9?
                Assistant: delegate_to_agent(agent_name='Division Agent', query='72 / 9', reason='query involves division')
                </example>
                
            </system>
            '''
        
    # Tool Function 
    async def delegate_to_agent(self, agent_name: str, query: str, reason: str) -> AgentDelegation:
        """
        Delegate the query to the appropriate agent.
        
        Args:
            agent_name: The name of the agent to delegate to
            query: The query to delegate
            reason: The reason for delegating to the agent
            
        Returns:
            An AgentDelegation object with the agent name, query, and reason
        """
        return AgentDelegation(agent_name=agent_name, query=query, reason=reason)
        
    async def clear_messages(self):
        """Clear the message history for this agent"""
        self.messages = []

########################################################
# Data Schema (Expected output from tool) (Add explicit tool names for strict validation)
########################################################
class NumberOperation(BaseModel):
    result: float
    operation: Literal['add_two_numbers', 'subtract_two_numbers', 'multiply_two_numbers', 'divide_two_numbers']
    numbers_used: List[float]

########################################################
# Addition Agent
########################################################
class AdditionAgent:
    aliases = ['addition', 'add', 'plus', '+']
    
    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = ollama.AsyncClient()
        
        # Define tools map as part of agent initialization
        self.tools_map = {
            'add_two_numbers': self.add_two_numbers,
        }
        
        self.messages = []
        self.available_tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_two_numbers",
                    "description": "Add two numbers, returning a NumberOperation.",
                    "parameters": {
                        "type": "object",
                        "required": ["a", "b"],
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        }
                    }
                }
            },
           
        ]
        
        # Validation Schema Map (Pydantic Model for Each Tool)
        self.validation_schema_map = {
            'add_two_numbers': NumberOperation,
        }
        
        # System Prompt (Dynaminc Tool Descriptions)
        self.system_prompt = f'''
            <system>
                You are an expert in addition that uses tools for calculations.
                
                <tools>
                {self.available_tools}
                </tools>
                
                <rules>
                1. ONLY call a tool when you have all required parameters
                2. If operation not available, respond: "Sorry, I can only perform addition."
                3. If parameters missing, ask for them
                4. For tool responses, extract ONLY the number from "result=X" for the final output
                </rules>
                
                <format>
                    <input>what is 5 plus 3?</input>
                    <tool_call>add_two_numbers(a=5, b=3)</tool_call>
                    <tool_response>result=8 operation='add_two_numbers' numbers_used=[5, 3]</tool_response>
                    <output>The result of 5 plus 3 is 8</output>
                </format>
                
                <instructions>
                DO NOT explain calculations.
                DO NOT make new tool calls after receiving a response.
                ONLY return the number from result=X. 
                DO NOT include the operation name or numbers used in the final output (e.g. "The result of 5 plus 3 is 8")
                </instructions>
                
                <example>
                User: what is 5 plus 3?
                Assistant: The result of 5 plus 3 is 8.
                </example>
                
              
            </system>
            '''
        
    # Tool Function 
    async def add_two_numbers(self, a: Any, b: Any) -> NumberOperation:
        """
        Add two numbers, but accept raw input (possibly string)
        and return a validated NumberOperation object.
        
        Args:
            a (Any): The first number
            b (Any): The second number
            
        Returns:
            NumberOperation: A validated NumberOperation object
        """
        return NumberOperation(result=float(a) + float(b), operation="add_two_numbers", numbers_used=[float(a), float(b)])
    
    async def clear_messages(self):
        """Clear the message history for this agent"""
        self.messages = []

########################################################
# Subtraction Agent
########################################################
class SubtractionAgent:
    aliases = ['subtraction', 'subtract', 'minus', '-']
    
    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = ollama.AsyncClient()
        
        # Define tools map as part of agent initialization
        self.tools_map = {
            'subtract_two_numbers': self.subtract_two_numbers,
        }

        self.messages = []
        self.available_tools = [
            
            {
                "type": "function",
                "function": {
                    "name": "subtract_two_numbers",
                    "description": "Subtract two numbers, returning a NumberOperation.",
                    "parameters": {
                        "type": "object",
                        "required": ["a", "b"],
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        }
                    }
                }
            },
            
        ]
        
        # Validation Schema Map (Pydantic Model for Each Tool)
        self.validation_schema_map = {
            'subtract_two_numbers': NumberOperation,
        }
        
        # System Prompt (Dynamic Tool Descriptions)
        self.system_prompt = f'''
            <system>
                You are an expert in subtraction that uses tools for calculations.
                
                <tools>
                {self.available_tools}
                </tools>
                
                <rules>
                1. ONLY call a tool when you have all required parameters
                2. If operation not available, respond: "Sorry, I can only perform subtraction."
                3. If parameters missing, ask for them
                4. For tool responses, extract ONLY the number from "result=X" for the final output
                </rules>
                
                <format>
                    <input>what is 5 minus 3?</input>
                    <tool_call>subtract_two_numbers(a=5, b=3)</tool_call>
                    <tool_response>result=2 operation='subtract_two_numbers' numbers_used=[5, 3]</tool_response>
                    <output>The result of 5 minus 3 is 2</output>
                </format>
                
                <instructions>
                DO NOT explain calculations.
                DO NOT make new tool calls after receiving a response.
                ONLY return the number from result=X. 
                DO NOT include the operation name or numbers used in the final output (e.g. "The result of 5 minus 3 is 2")
                </instructions>
                
                <example>
                User: what is 5 minus 3?
                Assistant: The result of 5 minus 3 is 2.
                </example>
                
            </system>
            '''
  
    # Tool Function 
    async def subtract_two_numbers(self, a: Any, b: Any) -> NumberOperation:
        """
        Subtract two numbers, returning a NumberOperation.
        
        Args:
            a (Any): The first number
            b (Any): The second number
            
        Returns:
            NumberOperation: A validated NumberOperation object
        """
        return NumberOperation(result=float(a) - float(b), operation="subtract_two_numbers", numbers_used=[float(a), float(b)])

    async def clear_messages(self):
        """Clear the message history for this agent"""
        self.messages = []

########################################################
# Multiplication Agent
########################################################
class MultiplicationAgent:
    aliases = ['multiplication', 'multiply', 'times', '*', 'x']
    
    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = ollama.AsyncClient()
        
        # Define tools map as part of agent initialization
        self.tools_map = {
            'multiply_two_numbers': self.multiply_two_numbers,
        }
        
        self.messages = []
        self.available_tools = [
            
            {
                "type": "function",
                "function": {
                    "name": "multiply_two_numbers",
                    "description": "Multiply two numbers, returning a NumberOperation.",
                    "parameters": {
                        "type": "object",
                        "required": ["a", "b"],
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        }
                    }
                }
            },
            
        ]
        
        # Validation Schema Map (Pydantic Model for Each Tool)
        self.validation_schema_map = {
            'multiply_two_numbers': NumberOperation,
        }
        
        # System Prompt (Dynamic Tool Descriptions)
        self.system_prompt = f'''
            <system>
                You are an expert in multiplication that uses tools for calculations.
                
                <tools>
                {self.available_tools}
                </tools>
                
                <rules>
                1. ONLY call a tool when you have all required parameters
                2. If operation not available, respond: "Sorry, I can only perform multiplication."
                3. If parameters missing, ask for them
                4. For tool responses, extract ONLY the number from "result=X" for the final output
                </rules>
                
                <format>
                    <input>what is 5 times 3?</input>
                    <tool_call>multiply_two_numbers(a=5, b=3)</tool_call>
                    <tool_response>result=15 operation='multiply_two_numbers' numbers_used=[5, 3]</tool_response>
                    <output>The result of 5 times 3 is 15</output>
                </format>
                
                <instructions>
                DO NOT explain calculations.
                DO NOT make new tool calls after receiving a response.
                ONLY return the number from result=X. 
                DO NOT include the operation name or numbers used in the final output (e.g. "The result of 5 times 3 is 15")
                </instructions>
                
                <example>
                User: what is 5 times 3?
                Assistant: The result of 5 times 3 is 15.
                </example>
               
            </system>
            '''
        
    
    async def multiply_two_numbers(self, a: Any, b: Any) -> NumberOperation:
        """
        Multiply two numbers, returning a NumberOperation.
        
        Args:
            a (Any): The first number
            b (Any): The second number
            
        Returns:
            NumberOperation: A validated NumberOperation object
        """
        return NumberOperation(result=float(a) * float(b), operation="multiply_two_numbers", numbers_used=[float(a), float(b)])
    
    async def clear_messages(self):
        """Clear the message history for this agent"""
        self.messages = []

########################################################
# Division Agent
########################################################
class DivisionAgent:
    aliases = ['division', 'divide', '/']
    
    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = ollama.AsyncClient()
        
        # Define tools map as part of agent initialization
        self.tools_map = {
            'divide_two_numbers': self.divide_two_numbers
        }
        
        self.messages = []
        self.available_tools = [
           
            {
                "type": "function",
                "function": {
                    "name": "divide_two_numbers",
                    "description": "Divide two numbers, returning a NumberOperation.",
                    "parameters": {
                        "type": "object",
                        "required": ["a", "b"],
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        }
                    }
                }
            }
        ]
        
        # Validation Schema Map (Pydantic Model for Each Tool)
        self.validation_schema_map = {
            'divide_two_numbers': NumberOperation
        }
        
        # System Prompt (Dynamic Tool Descriptions)
        self.system_prompt = f'''
            <system>
                You are an expert in division that uses tools for calculations.
                
                <tools>
                {self.available_tools}
                </tools>
                
                <rules>
                1. ONLY call a tool when you have all required parameters
                2. If operation not available, respond: "Sorry, I can only perform division."
                3. If parameters missing, ask for them
                4. For tool responses, extract ONLY the number from "result=X" for the final output
                </rules>
                
                <format>
                    <input>what is 5 divided by 3?</input>
                    <tool_call>divide_two_numbers(a=5, b=3)</tool_call>
                    <tool_response>result=1.6666666666666667 operation='divide_two_numbers' numbers_used=[5, 3]</tool_response>
                    <output>The result of 5 divided by 3 is 1.6666666666666667</output>
                </format>
                
                <instructions>
                DO NOT explain calculations.
                DO NOT make new tool calls after receiving a response.
                ONLY return the number from result=X. 
                DO NOT include the operation name or numbers used in the final output (e.g. "The result of 5 divided by 3 is 1.6666666666666667")
                </instructions>
                
                <example>
                User: what is 5 divided by 3?
                Assistant: The result of 5 divided by 3 is 1.6666666666666667.
                </example>
                
            </system>
            '''
  
    async def divide_two_numbers(self, a: Any, b: Any) -> NumberOperation:
        """
        Divide two numbers, returning a NumberOperation.
        
        Args:
            a (Any): The first number
            b (Any): The second number
            
        Returns:
            NumberOperation: A validated NumberOperation object
        """
        return NumberOperation(result=float(a) / float(b), operation="divide_two_numbers", numbers_used=[float(a), float(b)])

    async def clear_messages(self):
        """Clear the message history for this agent"""
        self.messages = []

########################################################
# Model Retry (Custom Retry Handler)
########################################################
class ModelRetryHandler:
    def __init__(
        self,
        client: AsyncClient,
        model: str,
        available_tools: List[Any],
        output_schema: Type[BaseModel],
        max_format_retries: int = 3,
        max_tool_retries: int = 2
    ):
        self.client = client
        self.model = model
        self.available_tools = available_tools
        self.output_schema = output_schema
        self.max_format_retries = max_format_retries
        self.max_tool_retries = max_tool_retries
        # Update tools_map to handle the tool definitions format
        self.tools_map = {
            tool['function']['name']: tool 
            for tool in available_tools 
            if isinstance(tool, dict) and 'function' in tool
        }

    async def _attempt_format_retry(
        self,
        raw_output: Any,
        expected_model: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """Attempts to reformat invalid output using the format parameter"""
        for attempt in range(self.max_format_retries):
            try:
                print(f'Format attempt {attempt + 1}/{self.max_format_retries}')
                format_messages = [
                    {
                        'role': 'system',
                        'content': 'You are a data formatter. Your only job is to convert the exact values from the input into the required schema format.'
                    },
                    {
                        'role': 'user',
                        'content': f'Convert this exact data into JSON matching this schema (preserve all values, just fix types): \n\n{expected_model.model_json_schema()}\n\nData to format: \n\n{raw_output}'
                    }
                ]
                
                # Send the formatted request to the model with the schema (Structured Output)
                retry_response = await self.client.chat(
                    self.model,
                    messages=format_messages,
                    format=expected_model.model_json_schema(),
                    options={'temperature': 0}
                )
                
                # Validate the response against the schema
                return expected_model.model_validate_json(retry_response.message.content)
                
            except ValidationError as e:
                print(f'Format attempt {attempt + 1} failed: {e}')
                if attempt == self.max_format_retries - 1:
                    return None
        return None

    # Tool Retry (Custom Retry Handler) (If simple formatting fails)
    async def _attempt_tool_retry(
        self,
        messages: List[dict],
        function_to_call: Any,
        output_schema: Type[BaseModel] = None
    ) -> Optional[BaseModel]:
        """Attempts to retry the entire tool call"""
        for attempt in range(self.max_tool_retries):
            try:
                print(f'Tool retry attempt {attempt + 1}/{self.max_tool_retries}')
                # Only use the system prompt and user query
                retry_messages = [
                    msg for msg in messages 
                    if msg['role'] in ('system', 'user')
                ][:2]
                
                tool_retry_response = await self.client.chat(
                    self.model,
                    messages=retry_messages,
                    tools=self.available_tools,
                    options={'temperature': 0}
                )
                
                if tool_retry_response.message.tool_calls:
                    retry_tool = tool_retry_response.message.tool_calls[0]
                    args = retry_tool.function.arguments
                    if isinstance(args, str):
                        args = json.loads(args)
                    retry_output = await function_to_call(**args)
                    return self.output_schema.model_validate(retry_output)
                    
            except (ValidationError, JSONDecodeError, Exception) as e:
                print(f'Tool retry attempt {attempt + 1} failed: {e}')
                if attempt == self.max_tool_retries - 1:
                    raise Exception(f"All retry attempts failed: {str(e)}")
        return None

    # Main Method (Execute Tool Call with Automatic Retries)
    async def execute_tool_with_retry(
        self,
        messages: List[dict],
        tool_call: Any,
        function_to_call: Any,
        output_schema: Type[BaseModel] = None
    ) -> tuple[BaseModel, List[dict]]:
        """Main method to execute a tool call with automatic retries"""
        try:
            # Use provided output_schema if available, otherwise fall back to default
            schema_to_use = output_schema or self.output_schema
            
            # Parse arguments if they're a string
            args = tool_call.function.arguments
            if isinstance(args, str):
                args = json.loads(args)

            # Await the function call
            raw_output = await function_to_call(**args)
            
            try:
                validated_output = schema_to_use.model_validate(raw_output)
                
                # Properly format the messages
                messages.extend([
                    {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [tool_call]
                    },
                    {
                        'role': 'tool',
                        'content': str(validated_output),
                        'name': tool_call.function.name
                    }
                ])
                
                return validated_output, messages
                
            except ValidationError:
                print('Validation error, attempting to reformat tool output')
                
                # Try format retry with the correct schema
                if validated_output := await self._attempt_format_retry(raw_output, schema_to_use):
                    messages.extend([
                        {
                            'role': 'assistant',
                            'content': None,
                            'tool_calls': [tool_call]
                        },
                        {
                            'role': 'tool',
                            'content': str(validated_output),
                            'name': tool_call.function.name
                        }
                    ])
                    return validated_output, messages
                
                # If format retry fails, try tool retry with the correct schema
                print('All formatting attempts failed, trying full tool retry')
                if validated_output := await self._attempt_tool_retry(messages, function_to_call, schema_to_use):
                    messages.extend([
                        {
                            'role': 'assistant',
                            'content': None,
                            'tool_calls': [tool_call]
                        },
                        {
                            'role': 'tool',
                            'content': str(validated_output),
                            'name': tool_call.function.name
                        }
                    ])
                    return validated_output, messages
                
                raise Exception("All retry attempts failed")
                
        except Exception as e:
            print(f"Error executing tool: {str(e)}")
            raise

########################################################
# Conversation Logger
########################################################
class ConversationLogger:
    def __init__(self, log_dir: Path = CONVERSATION_DIR):
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"conversation_log_{self.timestamp}.json"
        self.conversations = []
        
    def _serialize_message(self, message):
        """Helper method to serialize message content"""
        if isinstance(message, dict):
            return {
                "role": message.get("role", ""),
                "content": message.get("content", ""),
                "tool_calls": [self._serialize_tool_call(tc) for tc in message.get("tool_calls", [])] if message.get("tool_calls") else None
            }
        return str(message)
        
    def _serialize_tool_call(self, tool_call):
        """Helper method to serialize tool calls"""
        try:
            if hasattr(tool_call, 'function'):
                args = tool_call.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                return {
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": args
                    }
                }
            elif isinstance(tool_call, dict):
                return tool_call
            return {"type": "unknown", "content": str(tool_call)}
        except Exception as e:
            print(f"Error serializing tool call: {e}")
            return {"type": "error", "content": str(tool_call)}
        
    async def log_conversation(self, agent_name: str, messages: List[dict], response: Any, tool_calls: List[dict] = None):
        """Log a conversation entry to the JSON file"""
        try:
            # Serialize the response message
            response_message = {
                "model": getattr(response, 'model', None),
                "created_at": getattr(response, 'created_at', None),
                "message": {
                    "role": getattr(response.message, 'role', None),
                    "content": getattr(response.message, 'content', None),
                    "tool_calls": [
                        self._serialize_tool_call(tc) 
                        for tc in getattr(response.message, 'tool_calls', []) or []
                    ] if hasattr(response.message, 'tool_calls') else []
                } if hasattr(response, 'message') else {}
            }
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": agent_name,
                "messages": [self._serialize_message(msg) for msg in messages],
                "llm_response": response_message,
                "tool_executions": []
            }
            
            # Add any additional tool executions and their results
            if tool_calls:
                entry["tool_executions"].extend(tool_calls)
                
            self.conversations.append(entry)
            
            # Write to JSON file
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump({"conversations": self.conversations}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error logging conversation: {e}")
            
    async def log_tool_result(self, agent_name: str, tool_name: str, arguments: dict, result: Any):
        """Log a tool execution result"""
        try:
            if self.conversations:
                latest_conversation = self.conversations[-1]
                if latest_conversation["agent"] == agent_name:
                    tool_execution = {
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": str(result)
                    }
                    latest_conversation["tool_executions"].append(tool_execution)
                    
                    # Write updated conversations to file
                    with open(self.log_file, 'w', encoding='utf-8') as f:
                        json.dump({"conversations": self.conversations}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error logging tool result: {e}")

########################################################
# Run Agent with Tools Main LLM Call and Retry Handler
########################################################
async def run_agent_with_tools(
    client: AsyncClient,
    model: str,
    messages: List[dict],
    tools: List[dict],
    output_schema: Type[BaseModel],
    agent_instance: Any,
    stream: bool = True, 
    max_tool_retries: int = MAX_TOOL_RETRIES,
    agent_memory: bool = False,
    available_agents: Dict[str, Any] = None
) -> str:
    """
    Run an agent with tools and handle retries using ModelRetryHandler.
    
    Args:
        client: The AsyncClient instance
        model: The model to use for the agent
        messages: The list of messages to send to the agent
        tools: The list of tools to use
        output_schema: The output schema for the agent
        agent_instance: The agent instance to use
        stream: Whether to stream the response
        max_tool_retries: The maximum number of retries for tool calls
        agent_memory: If True, maintain agent's message history between runs.
                       If False, start fresh each time. Defaults to False.
    """
    # Clear the agent's message history before starting
    await agent_instance.clear_messages()
    
    # Initialize retry handler
    retry_handler = ModelRetryHandler(
        client=client,
        model=model,
        available_tools=tools,
        output_schema=output_schema,
        max_tool_retries=max_tool_retries
    )

    # Get agent name for logging
    agent_name = agent_instance.__class__.__name__.replace('Agent', '')
    
    # Handle message history based on agent_memory parameter
    if agent_memory:
        # Use existing messages if agent_memory is True
        agent_messages = messages
    else:
        # Create fresh message history if agent_memory is False
        agent_messages = [
            {'role': 'system', 'content': agent_instance.system_prompt},
            {'role': 'user', 'content': messages[-1]['content']} 
        ]
    
    # Get response from model
    response = await client.chat(
        model=model,
        messages=agent_messages,
        tools=tools,
        options={'temperature': 0}
    )

    # Log the conversation
    await conversation_logger.log_conversation(agent_name, agent_messages, response)

    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            print(f"[{agent_name}][TOOL] -> Using [{tool_call.function.name}] with args: {tool_call.function.arguments}\n")
            
            if tool_call.function.name == 'delegate_to_agent':
                args = tool_call.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                
                agent_name = args['agent_name']
                query = args['query']
                
                selected_agent = available_agents.get(agent_name)
                if not selected_agent:
                    return f"Error: Unknown agent {agent_name}"
                
                # Clear the selected agent's message history before use
                await selected_agent.clear_messages()
                
                # Create new message history for the selected agent
                agent_messages = [
                    {'role': 'system', 'content': selected_agent.system_prompt},
                    {'role': 'user', 'content': query}
                ]

                # Run the selected agent
                agent_response = await client.chat(
                    model=selected_agent.model,
                    messages=agent_messages,
                    tools=selected_agent.available_tools,
                    options={'temperature': 0}
                )
                
                # add the agent response to the conversation
                await conversation_logger.log_conversation(agent_name, agent_messages, agent_response)

                if agent_response.message.tool_calls:
                    tool_call = agent_response.message.tool_calls[0]
                    function_to_call = selected_agent.tools_map[tool_call.function.name]
                    
                    # Log the specialized agent's tool usage
                    print(f"[{agent_name}][TOOL] -> Using [{tool_call.function.name}] with args: {tool_call.function.arguments}")
                    
                    # Use retry handler for the actual calculation
                    validated_output, updated_messages = await retry_handler.execute_tool_with_retry(
                        messages=agent_messages,
                        tool_call=tool_call,
                        function_to_call=function_to_call,
                        output_schema=selected_agent.validation_schema_map[tool_call.function.name]
                    )

                    # Return result with agent name
                    if validated_output.result:
                        # run the agent again for the final response
                        final_response = await client.chat(
                            model=selected_agent.model,
                            messages=updated_messages,
                            tools=selected_agent.available_tools,
                            options={'temperature': 0}
                        )

                        # add the final response to the conversation
                        await conversation_logger.log_conversation(agent_name, updated_messages, final_response)
                        return f"[{agent_name}]: {final_response.message.content}"
                    else:
                        # add the final response to the conversation
                        await conversation_logger.log_conversation(agent_name, updated_messages, validated_output)
                        return f"[{agent_name}]: {validated_output.result}"
                
            else:
                # Handle direct tool calls (non-delegation)
                function_to_call = agent_instance.tools_map[tool_call.function.name]
                # Get the correct output schema from the validation_schema_map
                tool_output_schema = agent_instance.validation_schema_map[tool_call.function.name]
                validated_output, updated_messages = await retry_handler.execute_tool_with_retry(
                    messages=messages,
                    tool_call=tool_call,
                    function_to_call=function_to_call,
                    output_schema=tool_output_schema
                )
                
                if validated_output.result:
                    # run the agent again for the final response
                    final_response = await client.chat(
                    model=agent_instance.model,
                    messages=updated_messages,
                    tools=agent_instance.available_tools,
                    options={'temperature': 0}
                    )

                    # add the final response to the conversation
                    await conversation_logger.log_conversation(agent_name, updated_messages, final_response)
                    return f"[{agent_name}]: {final_response.message.content}" 
                else:
                    # add the final response to the conversation
                    await conversation_logger.log_conversation(agent_name, updated_messages, validated_output)
                    return f"[{agent_name}]: {validated_output.result}"
            
########################################################
# Agent Sequential Flows:
# Predefined Flows of Agents that can be executed
########################################################
class SequentialFlow:
    def __init__(self, name: str, steps: List[tuple], available_agents: Dict[str, Any]):
        """
        Initialize a sequential flow where each step's result feeds into the next step
        
        Args:
            name: Name of the flow
            steps: List of tuples (agent_class, operation_value)
                  e.g. [(AdditionAgent, 6), (SubtractionAgent, 3), ...]
        """
        self.name = name
        self.steps = [(agent(), value) for agent, value in steps]
        self.available_agents = available_agents
        
    async def execute(self, initial_value: float) -> float:
        current_value = initial_value
        print(f"\nExecuting {self.name} with initial value: {current_value}")
        print("----------------------------------------")
        
        for agent, value in self.steps:
            agent_name = agent.__class__.__name__.replace('Agent', '')
            print(f"Step: {agent_name} with {value}")
            
            # Create the appropriate query based on the agent type
            query = None
            if isinstance(agent, AdditionAgent):
                query = f"what is {current_value} plus {value}?"
            elif isinstance(agent, SubtractionAgent):
                query = f"what is {current_value} minus {value}?"
            elif isinstance(agent, MultiplicationAgent):
                query = f"what is {current_value} times {value}?"
            elif isinstance(agent, DivisionAgent):
                query = f"what is {current_value} divided by {value}?"
            
            if query is None:
                raise ValueError(f"Unknown agent type: {agent.__class__.__name__}")
            
            messages = [
                {'role': 'system', 'content': agent.system_prompt},
                {'role': 'user', 'content': query}
            ]
            
            output_schema = (
                agent.get_output_schema() 
                if hasattr(agent, 'get_output_schema') 
                else agent.validation_schema_map.get(
                    agent.available_tools[0]['function']['name']
                )
            )
            
            response = await run_agent_with_tools(
                client=agent.client,
                model=agent.model,
                messages=messages,
                tools=agent.available_tools,
                output_schema=output_schema,
                agent_instance=agent,
                available_agents=available_agents
            )
            
            # Extract the result and update current_value
            result_str = response.split("is ")[-1].strip(".")
            current_value = float(result_str)
            print(f"Result: {current_value}")
            print("----------------------------------------")
            
        return current_value


########################################################
#   Initialize available agents and flows
########################################################
available_agents = {
    'Addition Agent': AdditionAgent(),
    'Subtraction Agent': SubtractionAgent(),
    'Multiplication Agent': MultiplicationAgent(),
    'Division Agent': DivisionAgent()
}

# Define available flows
FLOWS = {
    "flow1": SequentialFlow(
        name="Basic Math Chain",
        steps=[
            (AdditionAgent, 10),      # Add 10
            (SubtractionAgent, 5),   # Subtract 5
            (MultiplicationAgent, 2), # Multiply by 2
            (DivisionAgent, 2),      # Divide by 2
        ],
        available_agents=available_agents
    ),
    "flow2": SequentialFlow(
        name="Double, Add, Halve",
        steps=[
            (MultiplicationAgent, 2), # Double it
            (AdditionAgent, 10),      # Add 10
            (DivisionAgent, 2),       # Halve it
        ],
        available_agents=available_agents
    )
}

########################################################
# Main 
########################################################
async def main():
    # Initialize Agents and Logger
    global conversation_logger
    conversation_logger = ConversationLogger()
    
    # Initialize Agents
    operator_agent = OperatorAgent()
    
    # Build alias map dynamically from agent aliases
    agent_alias_map = {}
    for agent in available_agents.values():
        for alias in agent.aliases:
            agent_alias_map[alias] = agent

    print("\n🦙 Ollama MAS Team! (type 'exit' to quit)")
    print("----------------------------------------")
    print("Available Agents:")
   
    # show agents and all their aliases
    for agent in available_agents.values():
        aliases_str = ', '.join([f'@{alias}' for alias in agent.aliases])
        print(f"  {agent.__class__.__name__.replace('Agent', '')}: ({aliases_str})")
    print("EXAMPLE: \"@add 10 and 4\"")
    print("----------------------------------------")
    print("\nAvailable Flows:")

    for name, flow in FLOWS.items():
        print(f"  @flow {name}: {flow.name}")
    print("EXAMPLE: \"@flow flow1 10\"")
    print("----------------------------------------")
    
    # Main Loop (CLI)
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            
            # Skip Empty Input
            if not user_input:
                continue
            
            # Add user input to conversation history
            conversation_history.append({'role': 'user', 'content': user_input})

            # Print Thinking Message
            print("\nAgent: Thinking...\n")
            
            # Clear operator agent's messages before each interaction
            await operator_agent.clear_messages()
            
            # Check for flow command (Run flow directly from CLI)
            if user_input.startswith('@flow'):
                parts = user_input.split()
                if len(parts) != 3:
                    print("Usage: @flow [name] [initial_value]")
                    continue
                    
                flow_name = parts[1].lower()
                try:
                    initial_value = float(parts[2])
                except ValueError:
                    print("Please provide a valid number for the initial value")
                    continue
                
                try:
                    results = await FLOWS[flow_name].execute(initial_value)
                    print("----------------------------------------")
                    print(f"Final result: {results}")
                except Exception as e:
                    print(f"Error executing flow: {str(e)}")
                continue
            
                
            # Check for direct agent command
            elif user_input.startswith('@'):
                parts = user_input[1:].split(maxsplit=1)
                if len(parts) == 2:
                    alias, query = parts
                    alias = alias.lower()

                    if alias in agent_alias_map:
                        selected_agent = agent_alias_map[alias]
                        messages = [
                            *conversation_history[-truncate_history:],
                            {'role': 'system', 'content': selected_agent.system_prompt},
                            {'role': 'user', 'content': query}
                        ]
                        
                        output_schema = (
                            selected_agent.get_output_schema() 
                            if hasattr(selected_agent, 'get_output_schema') 
                            else selected_agent.validation_schema_map.get(
                                selected_agent.available_tools[0]['function']['name']
                            )
                        )
                        
                        response = await run_agent_with_tools(
                            client=selected_agent.client,
                            model=selected_agent.model,
                            messages=messages,
                            tools=selected_agent.available_tools,
                            output_schema=output_schema,
                            agent_instance=selected_agent,
                            stream=True,
                            agent_memory=False,
                            available_agents=available_agents
                        )
                        
                        # Add the response to conversation history
                        conversation_history.append({'role': 'assistant', 'content': response})
                        
                        print(f"\nAgent: {response}")
                    else:
                        aliases = [f"@{alias}" for agent in available_agents.values() for alias in agent.aliases]
                        response = f"Unknown agent alias '@{alias}'. Available aliases: {', '.join(aliases)}"
                else:
                    response = "Invalid command format. Example: @add 5 and 4"
            else:
                # Regular delegation through operator agent
                messages = [
                    *conversation_history[-truncate_history:],
                    {'role': 'system', 'content': operator_agent.system_prompt},
                    {'role': 'user', 'content': user_input}
                ]
                
                response = await run_agent_with_tools(
                    client=operator_agent.client,
                    model=operator_agent.model,
                    messages=messages,
                    tools=operator_agent.available_tools,
                    output_schema=operator_agent.output_schema,
                    agent_instance=operator_agent,
                    stream=True,
                    agent_memory=True,
                    available_agents=available_agents
                )
            
            # Print Response (strip the agent prefix if present)
            cleaned_response = response.split(': ', 1)[-1] if ': ' in response else response
            print(f"\nAgent: {cleaned_response}")
            
            # Add the response to conversation history
            conversation_history.append({'role': 'assistant', 'content': cleaned_response})
        
            # print("\n----------------------------------------")
            # print("\nConversation History:")
            # for msg in conversation_history[-truncate_history:]:
            #     print(f"  {msg['role']}: {msg['content']}")
            # print("----------------------------------------")
                
        except ValidationError as e:
            print(f"\nValidation Error: {str(e)}")
            print("Please try your question again with valid numbers.")
            
        except Exception as e:
            print(f"\nAn unexpected error occurred: {str(e)}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    asyncio.run(main())