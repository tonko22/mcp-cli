import argparse
import logging
import sys
import anyio
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
    if command == "ping":
        print("Pinging Server...")
        result = await send_ping(read_stream, write_stream)
        print("Server is up and running" if result else "Server ping failed")
    elif command == "list-tools":
        print("Fetching Tools List...")
        tools = await send_tools_list(read_stream, write_stream)
        print("Tools List:", tools)
    elif command == "list-resources":
        print("Fetching Resources List...")
        resources = await send_resources_list(read_stream, write_stream)
        print("Resources List:", resources)
    elif command == "list-prompts":
        print("Fetching Prompts List...")
        prompts = await send_prompts_list(read_stream, write_stream)
        print("Prompts List:", prompts)
    else:
        print(f"Unknown command: {command}", file=sys.stderr)

async def main(config_path: str, server_name: str, command: str) -> None:
    """Main function to manage server initialization, communication, and shutdown."""
    process = None
    read_stream = None
    write_stream = None

    try:
        # Start the server process
        print("Starting Server Process", file=sys.stderr)

        # Load server configuration
        server_params = await load_config(config_path, server_name)

        # Launch server subprocess
        process = await anyio.open_process(
            [server_params.command, *server_params.args],
            env=server_params.env or get_default_environment(),
            stdin=True,
            stdout=True,
            stderr=sys.stderr,
        )

        # Establish stdio communication
        async with stdio_client(server_params) as (read_stream, write_stream):
            # Initialize the server
            init_result = await send_initialize(read_stream, write_stream)
            if not init_result:
                print("Server initialization failed", file=sys.stderr)
                return

            # Execute the specified command
            await handle_command(command, read_stream, write_stream)

        print("Shutting Down...", file=sys.stderr)

    except KeyboardInterrupt:
        print("KeyboardInterrupt received, initiating shutdown", file=sys.stderr)
    except Exception as e:
        print(f"Error in main: {e}", file=sys.stderr)
    finally:
        await shutdown_stdio_server(read_stream, write_stream, process)
        print("Shut Down!", file=sys.stderr)


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="MCP Command-Line Tool")

    # Add config-file argument
    parser.add_argument(
        "--config-file",
        default=DEFAULT_CONFIG_FILE,
        help="Path to the JSON configuration file containing server details.",
    )

    # Add server argument
    parser.add_argument(
        "--server",
        required=True,
        help="Name of the server configuration to use from the config file.",
    )

    # Add command argument
    parser.add_argument(
        "command",
        choices=["ping", "list-tools", "list-resources", "list-prompts"],
        help="Command to execute.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the main function using anyio's event loop
    try:
        anyio.run(main, args.config_file, args.server, args.command)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, exiting cleanly", file=sys.stderr)
