import os
from openai import OpenAI
from dotenv import load_dotenv
from system_prompt_generator import SystemPromptGenerator
from tools_handler import handle_tool_call, convert_to_openai_tools, fetch_tools

# Load environment variables
load_dotenv()

# Validate API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

async def handle_chat_mode(read_stream, write_stream):
    """Enter chat mode with multi-call support for autonomous tool chaining."""
    try:
        # get the tools from the server
        tools = await fetch_tools(read_stream, write_stream)

        # no tools
        if not tools:
            print("No tools available. Exiting chat mode.")
            return
        
        # generate the system prompt
        system_prompt = generate_system_prompt(tools)

        # convert tools to OpenAI format
        openai_tools = convert_to_openai_tools(tools)

        # create the client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # build the context
        conversation_history = [{"role": "system", "content": system_prompt}]

        # enter chat mode
        print("\nEntering chat mode. Type 'exit' to quit.")
        while True:
            user_message = input("\nYou: ").strip()
            if user_message.lower() in ["exit", "quit"]:
                print("Exiting chat mode.")
                break

            # add the item to the histroy
            conversation_history.append({"role": "user", "content": user_message})

            # process conversation
            await process_conversation(client, conversation_history, openai_tools, read_stream, write_stream)

    except Exception as e:
        print(f"\nError in chat mode: {e}")

def generate_system_prompt(tools):
    """Generate the system prompt for the assistant."""
    prompt_generator = SystemPromptGenerator()
    tools_json = {"tools": tools}

    # generate the system prompt
    system_prompt = prompt_generator.generate_prompt(tools_json)
    system_prompt += "\nReason step-by-step. If multiple steps are needed, call tools iteratively to achieve the goal. If unsure about data sources, use tools to describe them."
    
    # return the system prompt
    return system_prompt


async def process_conversation(client, conversation_history, openai_tools, read_stream, write_stream):
    """Process the conversation loop, handling tool calls and responses."""
    while True:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history,
            tools=openai_tools,
        )
        response_message = completion.choices[0].message

        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            print("Tool call response:", response_message.tool_calls)
            for tool_call in response_message.tool_calls:
                await handle_tool_call(tool_call, conversation_history, read_stream, write_stream)
        else:
            response_content = response_message.content
            print("Assistant:", response_content)
            conversation_history.append({"role": "assistant", "content": response_content})
            break
