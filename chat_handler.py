import os
import ollama
from openai import OpenAI
from dotenv import load_dotenv
from system_prompt_generator import SystemPromptGenerator
from messages.tools import send_tools_list

# Load environment variables
load_dotenv()

# Validate API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

async def handle_chat_mode(read_stream, write_stream):
    """Enter chat mode with the system prompt."""
    try:
        # Fetch tools dynamically
        print("\nFetching tools for chat mode...")

        # get the tools list from the server
        tools = await send_tools_list(read_stream, write_stream)

        # check we got tools
        if not tools:
            # failed to get tools
            print("Failed to fetch tools. Exiting chat mode.")
            return

        # Generate system prompt
        prompt_generator = SystemPromptGenerator()
        tools_json = {"tools": tools}
        system_prompt = prompt_generator.generate_prompt(tools_json)

        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print("\nEntering chat mode. Type 'exit' to quit.")
        while True:
            user_message = input("\nYou: ").strip()
            if user_message.lower() in ["exit", "quit"]:
                print("Exiting chat mode.")
                break

            # Prepare OpenAI chat messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            # Call OpenAI API for response
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            # # call ollama completion
            # completion = ollama.chat(
            #     'llama3.1',
            #     messages=messages
            # )

            # Access and print the response content
            response_content = completion.choices[0].message.content
            #response_content = completion.message.content
            print("Assistant:", response_content)

    except Exception as e:
        print(f"\nError in chat mode: {e}")
