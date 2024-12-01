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
    """
    Generate a comprehensive system prompt for the assistant.

    Args:
        tools (list): A list of tools available for the assistant.

    Returns:
        str: A system prompt string with detailed reasoning and problem-solving instructions.
    """
    prompt_generator = SystemPromptGenerator()
    tools_json = {"tools": tools}

    # Generate base prompt for tools
    system_prompt = prompt_generator.generate_prompt(tools_json)

    # Add autonomous and action-oriented reasoning guidelines
    system_prompt += """

SYSTEM PROMPT GUIDELINES

1. Take Decisive Actions:
   - Use available tools to explore schemas and dynamically construct appropriate queries.
   - Proceed autonomously without asking for user confirmation unless absolutely necessary.
   - Execute queries and adapt based on the results, providing concise summaries of actions taken.

2. Handle Tasks Systematically:
   - Start by identifying the structure of the database using schema exploration tools.
   - Dynamically adapt queries based on the discovered structure (e.g., table and column names).
   - Handle errors proactively by adjusting your approach and retrying without user intervention.

3. Use Tools Effectively:
   - Begin with tools to explore the schema if necessary information is not already known.
   - Construct queries based on real-time data and tool results rather than assumptions.
   - Avoid redundant tool calls and leverage prior results to optimize workflows.

4. Communicate Results Clearly:
   - Share a summary of the actions taken and the results obtained.
   - Focus on delivering outcomes, not intermediate explanations or queries.
   - Provide actionable next steps or insights based on the results.

5. Examples of Problem-Solving:
   - For database queries, dynamically discover schema information, construct appropriate queries, and execute them directly.
   - Handle missing resources or schema errors by adapting queries based on available data.

6. Make Logical Assumptions:
   - When specific details are not provided, apply common defaults:
     - Sort by `price` or `value` in descending order when ranking items.
     - Use standard relationships between tables (e.g., `products` linked to `sales` or `orders`).
   - Explain assumptions in the response but do not seek confirmation before acting.

EXAMPLES OF QUERY WORKFLOWS

User Query: "List all fields in the orders table."
- Step 1: Use tools to inspect the schema of the `orders` table.
- Step 2: Present the discovered fields concisely (e.g., `order_id`, `customer_id`, `order_date`).
- Step 3: Suggest next actions, such as filtering orders by date or customer.

REMEMBER
- Act autonomously and decisively, minimizing interruptions for user input.
- Use schema exploration tools to dynamically adapt to the database structure.
- Share results concisely and focus on outcomes rather than processes.

Return output in a user friend manner.
Let’s proceed efficiently and effectively!
"""

    return system_prompt

