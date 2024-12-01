import json
import re
from typing import Optional, Dict, Any
from messages.tools import call_tool, send_tools_list

def parse_tool_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse tool call from Llama's XML-style format."""
    function_regex = r"<function=(\w+)>(.*?)</function>"
    match = re.search(function_regex, response)
    
    if match:
        function_name, args_string = match.groups()
        try:
            args = json.loads(args_string)
            return {
                "function": function_name,
                "arguments": args,
            }
        except json.JSONDecodeError as error:
            print(f"Error parsing function arguments: {error}")
    return None

async def handle_tool_call(tool_call, conversation_history, read_stream, write_stream):
    """Handle a single tool call for both OpenAI and Llama formats."""
    # Handle direct OpenAI-style tool calls
    if hasattr(tool_call, 'function'):
        tool_name = tool_call.function.name
        raw_arguments = tool_call.function.arguments
    else:
        # Parse Llama's XML format from the last message
        last_message = conversation_history[-1]["content"]
        parsed_tool = parse_tool_response(last_message)
        if not parsed_tool:
            print("Unable to parse tool call from message")
            return
        tool_name = parsed_tool["function"]
        raw_arguments = parsed_tool["arguments"]

    try:
        # Parse the tool arguments
        tool_args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        print(f"Error decoding arguments for tool '{tool_name}': {raw_arguments}")
        return

    print(f"\nTool '{tool_name}' invoked with arguments: {tool_args}")

    # Execute the tool
    tool_response = await call_tool(tool_name, tool_args, read_stream, write_stream)
    if tool_response.get("isError"):
        print(f"Error calling tool: {tool_response.get('error')}")
        return

    # Format and display the response
    formatted_response = format_tool_response(tool_response.get("content", []))
    print(f"Tool '{tool_name}' Response:", formatted_response)

    # Add the tool response to the conversation history
    conversation_history.append(
        {"role": "assistant", "content": f"Tool '{tool_name}' Response: {formatted_response}"}
    )

def format_tool_response(response_content):
    """Format the response content from a tool."""
    if isinstance(response_content, list):
        return "\n".join(
            item.get("text", "No content") for item in response_content if item.get("type") == "text"
        )
    return str(response_content)

async def fetch_tools(read_stream, write_stream):
    """Fetch tools from the server."""
    print("\nFetching tools for chat mode...")

    # get the tools list
    tools_response = await send_tools_list(read_stream, write_stream)
    tools = tools_response.get("tools", [])
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        print("Invalid tools format received.")
        return None
    return tools

def convert_to_openai_tools(tools):
    """Convert tools into OpenAI-compatible function definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "parameters": tool.get("inputSchema", {}),
            },
        }
        for tool in tools
    ]