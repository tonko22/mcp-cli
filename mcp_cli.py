import argparse
import sys
import anyio
from config import load_config
from messages.send_initialize_message import send_initialize
from messages.send_ping_message import send_ping
from stdio_client import stdio_client, get_default_environment
from stdio_server_shutdown import shutdown_stdio_server

async def main(config_path: str, server_name: str) -> None:
    """ Main function to manage server initialization, communication, and shutdown. """
    process = None
    read_stream = None
    write_stream = None

    try:
        # starting
        print("Starting Server Process", file=sys.stderr)

        # Load server configuration
        server_params = await load_config(config_path, server_name)

        # Start the server process
        process = await anyio.open_process(
            [server_params.command, *server_params.args],
            env=server_params.env or get_default_environment(),
            stdin=True,
            stdout=True,
            stderr=sys.stderr,
        )

        # Establish stdio communication with the server
        async with stdio_client(server_params) as (read_stream, write_stream):
            # Initialize the server
            init_result = await send_initialize(read_stream, write_stream)

            # check if successful
            if not init_result:
                # failed
                print("Server initialization failed", file=sys.stderr)
                return

            # Send a ping message to the server
            print(f"Pinging Server...")
            result = await send_ping(read_stream, write_stream)

            # check the result
            if result is True:
                # success
                print("Server is up and running", file=sys.stderr)
            else:
                # failed
                print("Server ping failed", file=sys.stderr)

        # complete
        print("Shutting Down....", file=sys.stderr)

    except KeyboardInterrupt:
        # Handle KeyboardInterrupt
        print("KeyboardInterrupt received, initiating shutdown", file=sys.stderr)
    except Exception as e:
        # Handle other exceptions
        print(f"Error in main: {e}", file=sys.stderr)
    finally:
        # Perform server shutdown
        await shutdown_stdio_server(read_stream, write_stream, process)

        # complete
        print("Shut Down!", file=sys.stderr)


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="MCP Ping Client")

    # get the config
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to the JSON configuration file containing server details.",
    )

    # get the server name
    parser.add_argument(
        "--server",
        required=True,
        help="Name of the server configuration to use from the config file.",
    )

    # parse arguments
    args = parser.parse_args()

    # Run the main function using anyio's event loop
    try:
        anyio.run(main, args.config_file, args.server)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, exiting cleanly", file=sys.stderr)
