# send_ping.py
import logging
import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from messages.json_rpc_message import JSONRPCMessage

async def send_ping(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
) -> bool:
    """ Send a ping message to the server and log the response. """

    # construct the ping message
    message = JSONRPCMessage(id="ping-1", method="ping")

    # send the ping
    logging.debug("Sending ping message")
    await write_stream.send(message)

    try:
        # 5-second timeout for response
        with anyio.fail_after(5):  
            # get the response from the server
            async for response in read_stream:
                # if the response is an exception, log it and continue
                if isinstance(response, Exception):
                    logging.error(f"Error from server: {response}")
                    continue

                # log the response
                logging.debug(f"Server Response: {response.model_dump()}")

                # return if the message is a pong
                return True

    except TimeoutError:
        # log an error
        logging.error("Timeout waiting for ping response")
        return False
    except Exception as e:
        # log any other errors
        logging.error(f"Unexpected error during ping: {e}")
        return False
