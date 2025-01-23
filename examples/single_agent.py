import asyncio
from ollama import ChatResponse, AsyncClient
import ollama
from pydantic import BaseModel, ValidationError, validator
from typing import Any, List, Optional, Type, Literal
from json.decoder import JSONDecodeError
import json

# Ollama (https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)
# 3.2 model -> "3.2:3b+ supports function calling, but llama3.2:1b requires a tool call description added to the system prompt

# MODEL = 'llama3.2:1b'
MODEL = 'llama3.2:3b'
MAX_TOOL_RETRIES = 2 

# Data Schema (Expected output from tool) (Add explicit tool names for strict validation)
class NumberOperation(BaseModel):
    result: int
    operation: Literal['add_two_numbers', 'subtract_two_numbers', 'multiply_two_numbers', 'divide_two_numbers']
    numbers_used: list[int]

# Math Agent
class MathAgent:
    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = ollama.AsyncClient()
        
        # Define tools map as part of agent initialization
        self.tools_map = {
            'add_two_numbers': self.add_two_numbers,
            'subtract_two_numbers': self.subtract_two_numbers,
            'multiply_two_numbers': self.multiply_two_numbers,
            'divide_two_numbers': self.divide_two_numbers
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
            'add_two_numbers': NumberOperation,
            'subtract_two_numbers': NumberOperation,
            'multiply_two_numbers': NumberOperation,
            'divide_two_numbers': NumberOperation
        }
        
        # System Prompt (Dynamic Tool Descriptions)
        self.system_prompt = f'''
            <system>
                You are an expert in mathematical operations that uses tools for calculations.
                
                <tools>
                {self.available_tools}
                </tools>
                
                <rules>
                1. ONLY call a tool when you have all required parameters
                2. If operation not available, respond: "Sorry, I can only perform addition, subtraction, multiplication, and division."
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
                
                <example>
                User: what is 55 minus 3?
                Assistant: The result of 55 minus 3 is 52.
                </example>
                
                <example>
                User: what is 8 times 12?
                Assistant: The result of 8 times 12 is 96.
                </example>
                
                <example>
                User: what is 72 divided by 9?
                Assistant: The result of 72 divided by 9 is 8.
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
        return NumberOperation(result=int(a) + int(b), operation="add_two_numbers", numbers_used=[int(a), int(b)])
    
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
        return NumberOperation(result=int(a) - int(b), operation="subtract_two_numbers", numbers_used=[int(a), int(b)])
    
    async def multiply_two_numbers(self, a: Any, b: Any) -> NumberOperation:
        """
        Multiply two numbers, returning a NumberOperation.
        
        Args:
            a (Any): The first number
            b (Any): The second number
            
        Returns:
            NumberOperation: A validated NumberOperation object
        """
        return NumberOperation(result=int(a) * int(b), operation="multiply_two_numbers", numbers_used=[int(a), int(b)])
    
    async def divide_two_numbers(self, a: Any, b: Any) -> NumberOperation:
        """
        Divide two numbers, returning a NumberOperation.
        
        Args:
            a (Any): The first number
            b (Any): The second number
            
        Returns:
            NumberOperation: A validated NumberOperation object
        """
        return NumberOperation(result=int(a) / int(b), operation="divide_two_numbers", numbers_used=[int(a), int(b)])


async def run_agent_with_tools(
    client: AsyncClient,
    model: str,
    messages: List[dict],
    tools: List[dict],
    output_schema: Type[BaseModel],
    agent_instance: MathAgent,
    stream: bool = True, 
    max_tool_retries: int = MAX_TOOL_RETRIES
) -> str:
    """
    Run an agent with tools and handle retries.
    
    Args:
        client: Ollama AsyncClient instance
        model: Name of the model to use
        messages: List of conversation messages
        tools: List of tool definitions (JSON schema)
        output_schema: Pydantic model for validation
        max_tool_retries: Maximum number of tool retry attempts
    
    Returns:
        str: Final response from the model
    """
    
    response = await client.chat(
        model=model,
        messages=messages,
        tools=tools,
        options={'temperature': 0}
    )

    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = agent_instance.tools_map.get(function_name)
            
            if function_to_call:
                print(f'\n[TOOL] -> Using [{function_name}] with arguments: [{tool_call.function.arguments}]')
                
                # Try to execute and validate tool with retries
                for attempt in range(max_tool_retries):
                    try:
                        # Parse the arguments from string to dict if needed
                        args = tool_call.function.arguments
                        if isinstance(args, str):
                            args = json.loads(args)
                            
                        raw_output = await function_to_call(**args)
                        validated_output = output_schema.model_validate(raw_output)
                        
                        # Add tool call and validated output to messages
                        messages.append({
                            'role': 'assistant',
                            'content': None,
                            'tool_calls': [tool_call]
                        })
                        messages.append({
                            'role': 'tool',
                            'content': str(validated_output),
                            'name': function_name
                        })
                                                
                        break  # Exit retry loop on success
                        
                    except (ValidationError, JSONDecodeError) as e:
                        print(f"\nAttempt {attempt + 1} failed: {str(e)}")
                        if attempt == max_tool_retries - 1:  # Last attempt
                            raise  # Re-raise the last error if all retries failed
                        
                        # Add error feedback for retry context
                        messages.append([
                            {
                                'role': 'assistant',
                                'content': None,
                                'tool_calls': [tool_call]
                            },
                            {
                                'role': 'user',
                                'content': f"Please correct the function call. Errors encountered:\n{str(e)}"
                            }
                        ])
                        
                        
                        # Get new response for retry
                        response = await client.chat(
                            model=model,
                            messages=messages,
                            tools=tools,
                            options={'temperature': 0}
                        )
                        tool_call = response.message.tool_calls[0]
                                                
        # Stream the final response (LLM tool call interpretation)
        response_text = ""
        if stream:
            # Streaming response
            stream_response = await client.chat(
                model=model,
                messages=messages,
                tools=tools,
                options={'temperature': 0},
                stream=True
            )
            
            print("\nAGENT: ", end='', flush=True)
            async for part in stream_response:
                if part.message.content:
                    print(part.message.content, end='', flush=True)
                    response_text += part.message.content
            print()
        else:
            # Non-streaming response
            final_response = await client.chat(
                model=model,
                messages=messages,
                tools=tools,
                options={'temperature': 0}
            )
            response_text = final_response.message.content
            print(f"\nAGENT: {response_text}")
        
        return response_text

    return response.message.content


# Main 
async def main():
    # Initialize Math Agent
    agent = MathAgent()
    
    # Print Welcome Message
    print("\n🦙 Ollama Math Agent! (type 'exit' to quit)")
    print("----------------------------------------")
    
    while True:
        try:
            # Get User Input
            user_input = input("\nYou: ").strip()
            
            # Exit Condition
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            
            # Skip Empty Input
            if not user_input:
                continue
            
            # Print Thinking Message
            print("\nAGENT: Thinking...")
            
            # Create Message History Storage
            messages = [
                {'role': 'system', 'content': agent.system_prompt},
                {'role': 'user', 'content': user_input}
            ]

            # Run Agent with Tools
            response = await run_agent_with_tools(
                client=agent.client,
                model=agent.model,
                messages=messages,
                tools=agent.available_tools,
                output_schema=NumberOperation,
                agent_instance=agent,
                stream=True 
            )
            
            # Print Response (if not streamed)
            if not response.strip():
                print(f"\nAGENT: {response}")
            
        except ValidationError as e:
            # Print Error Message
            print("\nI apologize, but I encountered an error processing your calculation.")
            print("This might be due to an internal problem with my math tools.")
            print("Please try your question again, or try a different calculation.")
            
        except Exception as e:
            # Print Unexpected Error Message
            print(f"\nAn unexpected error occurred: {str(e)}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    asyncio.run(main())
