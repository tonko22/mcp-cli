import sys
import json
import logging
import anyio
from anyio.streams.text import TextReceiveStream
from contextlib import asynccontextmanager
from environment import get_default_environment
from messages.json_rpc_message import JSONRPCMessage
from stdio_server_parameters import StdioServerParameters

# Configure logging with timestamps and log levels for better debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

@asynccontextmanager
async def stdio_client(server: StdioServerParameters):
    """ Asynchronous context manager for a stdio-based client-server communication. """
    # Validate server command and arguments
    if not server.command:
        raise ValueError("Server command must not be empty.")
    if not isinstance(server.args, (list, tuple)):
        raise ValueError("Server arguments must be a list or tuple.")

    # Create memory streams for read and write operations
    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    # Open the subprocess with the given command and environment
    process = await anyio.open_process(
        [server.command, *server.args],
        env=server.env or get_default_environment(),
        stderr=sys.stderr,
    )

    async def stdout_reader():
        """ Reads JSON-RPC messages from the server's stdout and sends them to the read stream. """
        assert process.stdout, "Opened process is missing stdout"

        # Initialize an empty buffer for partial lines
        buffer = ""

        try:
            # use the read stream writer
            async with read_stream_writer:
                # reach each chunk from stdout until it's closed
                async for chunk in TextReceiveStream(process.stdout):
                    # Split the incoming chunk into lines and process each line
                    lines = (buffer + chunk).split("\n")

                    # Save the last partial line to buffer
                    buffer = lines.pop()  
                    for line in lines:
                        # Skip empty lines
                        if not line.strip():  
                            continue

                        try:
                            # Parse and validate JSON-RPC message
                            data = json.loads(line)

                            # Validate the parsed data against the JSON-RPC schema
                            message = JSONRPCMessage.model_validate(data)

                            # Send the validated message to the read stream
                            await read_stream_writer.send(message)
                        except json.JSONDecodeError as exc:
                            # Log JSON decoding errors
                            logging.error(f"JSON decode error: {exc}. Line: {line.strip()}")
                        except Exception as exc:
                            # Log other errors
                            logging.error(f"Error processing message: {exc}. Line: {line.strip()}")
                
                # Handle any remaining data in the buffer after the stream ends
                if buffer.strip():
                    try:
                        # load the remaining data in the buffer
                        data = json.loads(buffer)

                        # validate the parsed data against the JSON-RPC schema
                        message = JSONRPCMessage.model_validate(data)

                        # send the validated message to the read stream
                        await read_stream_writer.send(message)
                    except json.JSONDecodeError as exc:
                        # Log JSON decoding errors
                        logging.error(f"JSON decode error for leftover buffer: {exc}. Buffer: {buffer.strip()}")
                    except Exception as exc:
                        # Log other errors
                        logging.error(f"Error processing leftover buffer: {exc}. Buffer: {buffer.strip()}")
        except anyio.ClosedResourceError:
            # closed
            logging.debug("Read stream closed.")
        except Exception as exc:
            # error
            logging.error(f"Unexpected error in stdout_reader: {exc}")

    async def stdin_writer():
        """ Sends JSON-RPC messages from the write stream to the server's stdin. """
        assert process.stdin, "Opened process is missing stdin"
        try:
            async with write_stream_reader:
                async for message in write_stream_reader:
                    try:
                        # Serialize the message to a JSON string
                        json_str = message.model_dump_json(exclude_none=True)
                        logging.debug(f"Sending: {json_str}")
                        # Send the message to the server's stdin
                        await process.stdin.send((json_str + "\n").encode())
                    except Exception as exc:
                        logging.error(f"Error writing to stdin: {exc}")
        except anyio.ClosedResourceError:
            logging.debug("Write stream closed.")
        except anyio.TimeoutError:
            logging.error("Timeout while writing to stdin.")
        except Exception as exc:
            logging.error(f"Unexpected error in stdin_writer: {exc}")

    async def terminate_process():
        """
        Gracefully terminates the subprocess.
        
        Ensures the process is terminated and resources are cleaned up. Logs
        any errors during termination.
        """
        try:
            process.terminate()
            await process.wait()
        except Exception as exc:
            logging.error(f"Error terminating process: {exc}")
            try:
                logging.warning("Forcefully killing the process.")
                process.kill()
            except Exception as kill_exc:
                logging.error(f"Error killing process: {kill_exc}")

    try:
        async with anyio.create_task_group() as tg, process:
            # Start the reader and writer tasks in parallel
            tg.start_soon(stdout_reader)
            tg.start_soon(stdin_writer)

            # Yield the read and write streams for the caller to use
            yield read_stream, write_stream

        # Wait for the process to exit and log its exit code
        exit_code = await process.wait()
        logging.debug(f"Process exited with code {exit_code}")
    except Exception as exc:
        logging.error(f"Unexpected error during stdio_client execution: {exc}")
    finally:
        # Ensure the process is terminated on exit
        await terminate_process()
