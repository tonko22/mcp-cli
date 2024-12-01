from llm_client import LLMClient
from tools_handler import handle_tool_call, convert_to_openai_tools, fetch_tools
from system_prompt_generator import SystemPromptGenerator

async def handle_chat_mode(read_stream, write_stream, provider="openai"):
    """Enter chat mode with multi-call support for autonomous tool chaining."""
    try:
        # Fetch tools dynamically
        tools = await fetch_tools(read_stream, write_stream)

        if not tools:
            print("No tools available. Exiting chat mode.")
            return

        # Generate system prompt
        system_prompt = generate_system_prompt(tools)

        # Convert tools to OpenAI format (only relevant for OpenAI)
        openai_tools = convert_to_openai_tools(tools) if provider == "openai" else None

        # Initialize the LLM client
        client = LLMClient(provider=provider)

        conversation_history = [{"role": "system", "content": system_prompt}]

        print("\nEntering chat mode. Type 'exit' to quit.")
        while True:
            user_message = input("\nYou: ").strip()
            if user_message.lower() in ["exit", "quit"]:
                print("Exiting chat mode.")
                break

            conversation_history.append({"role": "user", "content": user_message})

            # Process conversation
            await process_conversation(client, conversation_history, openai_tools, read_stream, write_stream)

    except Exception as e:
        print(f"\nError in chat mode: {e}")


async def process_conversation(client, conversation_history, openai_tools, read_stream, write_stream):
    """Process the conversation loop, handling tool calls and responses."""
    while True:
        # Call the LLM client
        completion = client.create_completion(
            messages=conversation_history,
            tools=openai_tools,
        )

        response_content = completion.get("response", "No response")
        tool_calls = completion.get("tool_calls", [])

        # If tool calls are present, process them
        if tool_calls:
            for tool_call in tool_calls:
                await handle_tool_call(tool_call, conversation_history, read_stream, write_stream)
            continue  # Continue the loop to handle follow-up responses

        # Otherwise, process as a regular assistant response
        print("Assistant:", response_content)
        conversation_history.append({"role": "assistant", "content": response_content})
        break



def generate_system_prompt(tools):
    """Generate the system prompt for the assistant."""
    prompt_generator = SystemPromptGenerator()
    tools_json = {"tools": tools}

    # generate the system prompt
    system_prompt = prompt_generator.generate_prompt(tools_json)
    system_prompt += "\nReason step-by-step. If multiple steps are needed, call tools iteratively to achieve the goal. If unsure about data sources, use tools to describe them."
    
    # return the system prompt
    return system_prompt