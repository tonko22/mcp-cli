# chat_handler.py
from llm_client import LLMClient
from tools_handler import handle_tool_call, convert_to_openai_tools, fetch_tools, parse_tool_response
from system_prompt_generator import SystemPromptGenerator

async def handle_chat_mode(read_stream, write_stream, provider="openai"):
    """Enter chat mode with multi-call support for autonomous tool chaining."""
    try:
        # fetch tools dynamically
        tools = await fetch_tools(read_stream, write_stream)

        if not tools:
            print("No tools available. Exiting chat mode.")
            return

        # generate system prompt
        system_prompt = generate_system_prompt(tools)

        # convert tools to OpenAI format
        openai_tools = convert_to_openai_tools(tools)

        # Initialize the LLM client
        client = LLMClient(provider=provider)

        # setup the conversation history
        conversation_history = [{"role": "system", "content": system_prompt}]

        # entering chat mode
        print("\nEntering chat mode. Type 'exit' to quit.")
        while True:
            try:
                user_message = input("\nYou: ").strip()
                if user_message.lower() in ["exit", "quit"]:
                    print("Exiting chat mode.")
                    break

                # add user message to history
                conversation_history.append({"role": "user", "content": user_message})

                # Process conversation
                await process_conversation(client, conversation_history, openai_tools, read_stream, write_stream)

            except Exception as e:
                print(f"\nError processing message: {e}")
                continue

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
            # loop through tool calls
            for tool_call in tool_calls:
                await handle_tool_call(tool_call, conversation_history, read_stream, write_stream)

            # Continue the loop to handle follow-up responses
            continue  

        # Otherwise, process as a regular assistant response
        print("Assistant:", response_content)
        conversation_history.append({"role": "assistant", "content": response_content})
        break
        

def generate_system_prompt(tools):
    """Generate a comprehensive system prompt for the assistant."""
    prompt_generator = SystemPromptGenerator()
    tools_json = {"tools": tools}

    # Get base prompt for tools
    system_prompt = prompt_generator.generate_prompt(tools_json)

    # Add general reasoning and problem-solving guidelines
    system_prompt += """

REASONING APPROACH:
1. Think step-by-step through each task
2. Break complex problems into smaller parts
3. Validate assumptions before proceeding
4. Learn from results and adjust approach

USING TOOLS:
1. Start with exploration
   - Check available information
   - Understand data structure
   - Verify assumptions
   
2. Work iteratively
   - Begin with simple queries
   - Build on successful results
   - Adjust based on what you learn

3. Handle errors productively
   - Analyze error messages carefully
   - Use errors as guidance
   - Try alternative approaches
   - Explain what went wrong

4. Communicate clearly
   - Explain your thinking
   - Share what you discover
   - Describe next steps
   - Ask for clarification when needed

EXAMPLES:
- If working with databases:
  * Check schema before querying
  * Verify column names exist
  * Start with simple queries
  
- If processing data:
  * Verify data format first
  * Handle edge cases
  * Validate results

- If accessing resources:
  * Check availability
  * Verify permissions
  * Handle missing data

Remember: Be systematic, thorough, and clear in your explanations. Each tool call should have a clear purpose that you explain to the user."""

    return system_prompt