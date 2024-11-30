import logging
import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from typing import Optional


async def shutdown_stdio_server(
    read_stream: Optional[MemoryObjectReceiveStream],
    write_stream: Optional[MemoryObjectSendStream],
    process: anyio.abc.Process,
    timeout: float = 5.0,
) -> None:
    """
    Gracefully shutdown a stdio-based server.

    This function performs the following steps:
    1. Closes the stdin stream of the process.
    2. Waits for the process to terminate gracefully.
    3. Sends SIGTERM if the process does not terminate within the timeout.
    4. Sends SIGKILL if the process does not terminate after SIGTERM.
    5. Logs each step and ensures cleanup in case of errors.

    Args:
        read_stream (Optional[MemoryObjectReceiveStream]): Stream to receive responses.
        write_stream (Optional[MemoryObjectSendStream]): Stream to send requests.
        process (anyio.abc.Process): The server process.
        timeout (float): Time to wait for graceful shutdown and SIGTERM before escalation.
    """
    logging.info("Initiating stdio server shutdown")

    try:
        # Step 1: Close the write stream (stdin for the server)
        if process.stdin:
            await process.stdin.aclose()
            logging.info("Closed stdin stream")

        # Step 2: Wait for the process to terminate gracefully
        with anyio.fail_after(timeout):
            await process.wait()
            logging.info("Process exited normally")
            return

    except TimeoutError:
        logging.warning(f"Server did not exit within {timeout} seconds, sending SIGTERM")
        process.terminate()

        try:
            # Step 3: Wait for the process to terminate after SIGTERM
            with anyio.fail_after(timeout):
                await process.wait()
                logging.info("Process exited after SIGTERM")
                return
        except TimeoutError:
            logging.warning("Server did not respond to SIGTERM, sending SIGKILL")
            process.kill()

            # Step 4: Wait for the process to terminate after SIGKILL
            await process.wait()
            logging.info("Process exited after SIGKILL")

    except Exception as e:
        # Catch unexpected errors during shutdown
        logging.error(f"Unexpected error during stdio server shutdown: {e}")
        process.kill()
        await process.wait()
        logging.info("Process forcibly terminated")

    finally:
        logging.info("Stdio server shutdown complete")
