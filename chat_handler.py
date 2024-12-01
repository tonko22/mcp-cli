import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from system_prompt_generator import SystemPromptGenerator
from messages.tools import send_tools_list, call_tool

# Load environment variables
load_dotenv()

# Validate API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

async def handle_chat_mode(read_stream, write_stream):
    """Enter chat mode with multi-call support for autonomous tool chaining."""
    try:
        # Fetch tools dynamically
        print("\nFetching tools for chat mode...")
        tools_response = await send_tools_list(read_stream, write_stream)

        # Extract tools list
        tools = tools_response.get("tools", [])

        # Validate tools structure
        if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
            print("Invalid tools format received. Expected a list of dictionaries.")
            return

        # Generate system prompt with CoT
        prompt_generator = SystemPromptGenerator()
        tools_json = {"tools": tools}
        system_prompt = prompt_generator.generate_prompt(tools_json)
        system_prompt += "\nReason step-by-step. If multiple steps are needed, call tools iteratively to achieve the goal.  if you are unsure the schema of data sources, you can check if there is a tool to describe a data source"
        # Convert tools into OpenAI-compatible function definitions
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "parameters": tool.get("inputSchema", {}),
                },
            }
            for tool in tools
        ]

        # Debugging: Print OpenAI tools configuration
        print("Configured OpenAI tools:", openai_tools)

        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print("\nEntering chat mode. Type 'exit' to quit.")
        conversation_history = [{"role": "system", "content": system_prompt}]

        while True:
            user_message = input("\nYou: ").strip()
            if user_message.lower() in ["exit", "quit"]:
                print("Exiting chat mode.")
                break

            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": user_message})

            # Loop for iterative tool calls or assistant responses
            while True:
                # Call OpenAI API with tools
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=conversation_history,
                    tools=openai_tools,
                )

                # Access the response or tool call
                response_message = completion.choices[0].message

                if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                    # Debugging: Print the tool call response
                    print("Tool call response:", response_message.tool_calls)

                    # Handle tool call response
                    for tool_call in response_message.tool_calls:
                        tool_name = tool_call.function.name
                        raw_arguments = tool_call.function.arguments
                        try:
                            tool_args = json.loads(raw_arguments) if raw_arguments else {}
                        except json.JSONDecodeError:
                            print(f"Error decoding arguments for tool '{tool_name}': {raw_arguments}")
                            continue

                        print(f"\nTool '{tool_name}' invoked with arguments: {tool_args}")

                        # Call the tool using the provided arguments
                        tool_response = await call_tool(tool_name, tool_args, read_stream, write_stream)
                        if tool_response.get("isError"):
                            print(f"Error calling tool: {tool_response.get('error')}")
                            break

                        # Process and format tool response
                        response_content = tool_response.get("content", [])
                        formatted_response = ""
                        if isinstance(response_content, list):
                            for item in response_content:
                                if item.get("type") == "text":
                                    formatted_response += item.get("text", "No content") + "\n"
                        else:
                            formatted_response = str(response_content)

                        print(f"Tool '{tool_name}' Response:", formatted_response)

                        # Add tool response to conversation history
                        conversation_history.append(
                            {"role": "assistant", "content": f"Tool '{tool_name}' Response: {formatted_response}"}
                        )
                else:
                    # Handle normal assistant response
                    response_content = response_message.content
                    print("Assistant:", response_content)
                    conversation_history.append({"role": "assistant", "content": response_content})
                    break

    except Exception as e:
        print(f"\nError in chat mode: {e}")
