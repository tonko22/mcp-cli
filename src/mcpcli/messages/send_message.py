# messages/send_message.py
import logging
import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcpcli.messages.message_types.json_rpc_message import JSONRPCMessage

async def send_message(
    read_stream: MemoryObjectReceiveStream,
    write_stream: MemoryObjectSendStream,
    message: JSONRPCMessage,
    timeout: float = 5,
    retries: int = 3,
) -> dict:
    """
    Send a JSON-RPC message to the server and return the response.

    Args:
        read_stream (MemoryObjectReceiveStream): The stream to read responses.
        write_stream (MemoryObjectSendStream): The stream to send requests.
        message (JSONRPCMessage): The JSON-RPC message to send.
        timeout (float): Timeout in seconds to wait for a response.
        retries (int): Number of retry attempts.

    Returns:
        dict: The server's response as a dictionary.

    Raises:
        TimeoutError: If no response is received within the timeout.
        Exception: If an unexpected error occurs.
    """
    for attempt in range(1, retries + 1):
        try:
            logging.debug(f"Attempt {attempt}/{retries}: Sending message: {message}")
            logging.debug(f"Message details - method: {message.method}, params: {message.params}")
            await write_stream.send(message)

            with anyio.fail_after(timeout):
                async for response in read_stream:
                    if not isinstance(response, Exception):
                        # Проверяем наличие ошибки в ответе
                        if response.error is not None:
                            error_msg = response.error.get("message", str(response.error))
                            logging.error(f"JSON-RPC error: {error_msg}")
                            raise ValueError(error_msg)
                        
                        # Если есть result, используем его
                        if response.result is not None:
                            logging.debug(f"Received result: {response.result}")
                            return {"result": response.result}
                            
                        # Если нет result, но есть другие поля, возвращаем их
                        response_data = response.model_dump(exclude_none=True)
                        logging.debug(f"Received response: {response_data}")
                        return response_data
                    else:
                        logging.error(f"Server error: {str(response)}")
                        logging.error(f"Error type: {type(response)}")
                        logging.error(f"Error details: {getattr(response, '__dict__', {})}")
                        raise response

        except TimeoutError:
            logging.error(
                f"Timeout waiting for response to message '{message.method}' (Attempt {attempt}/{retries})"
            )
            if attempt == retries:
                raise
        except Exception as e:
            logging.error(
                f"Unexpected error during '{message.method}' request: {str(e)} (Attempt {attempt}/{retries})"
            )
            logging.error(f"Error type: {type(e)}")
            logging.error(f"Error details: {getattr(e, '__dict__', {})}")
            if attempt == retries:
                raise

        await anyio.sleep(2)
