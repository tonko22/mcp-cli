import argparse
import logging
import sys
import anyio
import asyncio
import os
from config import load_config
from messages.tools import send_tools_list
from messages.resources import send_resources_list
from messages.prompts import send_prompts_list
from messages.send_initialize_message import send_initialize
from messages.send_ping_message import send_ping
from stdio_client import stdio_client, get_default_environment
from stdio_server_shutdown import shutdown_stdio_server

# Default path for the configuration file
DEFAULT_CONFIG_FILE = "server_config.json"

async def handle_command(command: str, read_stream, write_stream):
    """Handle specific commands dynamically."""
    try:
        if command == "ping":
            print("\nPinging Server...")
            result = await send_ping(read_stream, write_stream)
            print("Server is up and running" if result else "Server ping failed")
        elif command == "list-tools":
            print("\nFetching Tools List...")
            tools = await send_tools_list(read_stream, write_stream)
            print("Tools List:", tools)
        elif command == "list-resources":
            print("\nFetching Resources List...")
            resources = await send_resources_list(read_stream, write_stream)
            print("Resources List:", resources)
        elif command == "list-prompts":
            print("\nFetching Prompts List...")
            prompts = await send_prompts_list(read_stream, write_stream)
            print("Prompts List:", prompts)
        elif command in ["quit", "exit"]:
            print("\nGoodbye!")
            return False
        elif command == "clear":
            if sys.platform == "win32":
                os.system("cls")
            else:
                os.system("clear")
        elif command == "help":
            print("\nAvailable commands:")
            print("  ping          - Check if server is responsive")
            print("  list-tools    - Display available tools")
            print("  list-resources- Display available resources")
            print("  list-prompts  - Display available prompts")
            print("  clear         - Clear the screen")
            print("  help          - Show this help message")
            print("  quit/exit     - Exit the program")
        else:
            print(f"\nUnknown command: {command}")
            print("Type 'help' for available commands")
    except Exception as e:
        print(f"\nError executing command: {e}")
    
    return True

async def get_input():
    """Get input asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input("\n> ").strip().lower())

async def interactive_mode(read_stream, write_stream):
    """Run the CLI in interactive mode."""
    print("\nWelcome to the Interactive MCP Command-Line Tool")
    print("Type 'help' for available commands or 'quit' to exit")
    
    while True:
        try:
            command = await get_input()
            if not command:
                continue
                
            should_continue = await handle_command(command, read_stream, write_stream)
            if not should_continue:
                return
                
        except KeyboardInterrupt:
            print("\nUse 'quit' or 'exit' to close the program")
        except EOFError:
            break
        except Exception as e:
            print(f"\nError: {e}")

class GracefulExit(Exception):
    """Custom exception for handling graceful exits."""
    pass

async def main(config_path: str, server_name: str, command: str = None) -> None:
    """Main function to manage server initialization, communication, and shutdown."""
    try:
        # Load server configuration
        server_params = await load_config(config_path, server_name)
        
        # Establish stdio communication
        async with stdio_client(server_params) as (read_stream, write_stream):
            # Initialize the server
            init_result = await send_initialize(read_stream, write_stream)
            if not init_result:
                print("Server initialization failed")
                return

            if command:
                # Single command mode
                await handle_command(command, read_stream, write_stream)
            else:
                # Interactive mode
                await interactive_mode(read_stream, write_stream)
                
            # Break the event loop to ensure clean exit
            loop = asyncio.get_event_loop()
            loop.stop()

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Ensure we exit the process
        os._exit(0)

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="MCP Command-Line Tool")

    parser.add_argument(
        "--config-file",
        default=DEFAULT_CONFIG_FILE,
        help="Path to the JSON configuration file containing server details.",
    )

    parser.add_argument(
        "--server",
        required=True,
        help="Name of the server configuration to use from the config file.",
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["ping", "list-tools", "list-resources", "list-prompts"],
        help="Command to execute (optional - if not provided, enters interactive mode).",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )

    try:
        anyio.run(main, args.config_file, args.server, args.command)
    except KeyboardInterrupt:
        os._exit(0)
    except Exception:
        os._exit(1)