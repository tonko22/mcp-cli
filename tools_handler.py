import json
from messages.tools import call_tool, send_tools_list

import json

async def handle_tool_call(tool_call, conversation_history, read_stream, write_stream):
    """Handle a single tool call."""
    # Validate tool call structure
    if not hasattr(tool_call, "function") or not hasattr(tool_call.function, "name"):
        print("Invalid tool call: Missing function or name.")
        conversation_history.append(
            {"role": "assistant", "content": "Error: Invalid tool call structure."}
        )
        return

    tool_name = tool_call.function.name
    raw_arguments = getattr(tool_call.function, "arguments", "{}")

    # Parse tool arguments
    try:
        tool_args = json.loads(raw_arguments) if raw_arguments else {}
    except json.JSONDecodeError:
        error_message = f"Error decoding arguments for tool '{tool_name}': {raw_arguments}"
        print(error_message)
        conversation_history.append({"role": "assistant", "content": error_message})
        return

    print(f"\nTool '{tool_name}' invoked with arguments: {tool_args}")

    # Execute the tool
    try:
        tool_response = await call_tool(tool_name, tool_args, read_stream, write_stream)
    except Exception as e:
        error_message = f"Error executing tool '{tool_name}': {str(e)}"
        print(error_message)
        conversation_history.append({"role": "assistant", "content": error_message})
        return

    if tool_response.get("isError"):
        error_message = f"Error calling tool '{tool_name}': {tool_response.get('error')}"
        print(error_message)
        conversation_history.append({"role": "assistant", "content": error_message})
        return

    # Format and display the response
    try:
        formatted_response = format_tool_response(tool_response.get("content", []))
    except Exception as e:
        formatted_response = f"Error formatting tool response: {str(e)}"
        print(formatted_response)

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