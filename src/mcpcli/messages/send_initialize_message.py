# messages/send_initialize_message.py
import logging
import anyio
from typing import Optional
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from mcpcli.messages.message_types.initialize_message import (
    InitializeMessage,
    InitializedNotificationMessage,
    InitializeParams,
    MCPClientCapabilities,
    MCPClientInfo,
    InitializeResult,
)


async def send_initialize(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
) -> Optional[InitializeResult]:
    """Send an initialization request to the server and process its response."""

    # Set initialize params
    init_params = InitializeParams(
        protocolVersion="2024-11-05",
        capabilities=MCPClientCapabilities(),
        clientInfo=MCPClientInfo(),
    )

    # Create the initialize message
    init_message = InitializeMessage(init_params)

    # Sending
    logging.info("Sending initialize request with params: %s", init_params.model_dump())
    await write_stream.send(init_message)

    try:
        # 15-second timeout for response
        timeout = 15
        logging.info(f"Waiting for server initialization response (timeout: {timeout}s)")
        with anyio.fail_after(timeout):
            # Get the response from the server
            async for response in read_stream:
                # If the response is an exception, log it and continue
                if isinstance(response, Exception):
                    logging.error("Error from server during initialization: %s", str(response), exc_info=True)
                    continue

                # Debug log the received message
                logging.debug("Received initialization response: %s", response.model_dump())

                # Check for error
                if response.error:
                    logging.error("Server returned error during initialization: %s", response.error)
                    return None

                # Check for result
                if response.result:
                    try:
                        # Validate the result
                        init_result = InitializeResult.model_validate(response.result)
                        logging.debug("Server initialized successfully")

                        # Notify the server of successful initialization
                        initialized_notify = InitializedNotificationMessage()
                        await write_stream.send(initialized_notify)

                        return init_result
                    except Exception as e:
                        logging.error(f"Error processing init result: {e}")
                        return None

    except TimeoutError:
        logging.error("Timeout waiting for server initialization response")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during server initialization: {e}")
        raise

    # Timeout
    logging.error("Initialization response timeout")
    return None
