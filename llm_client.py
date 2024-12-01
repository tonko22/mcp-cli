# llm_client.py
import os
import json
import ollama
from openai import OpenAI
from dotenv import load_dotenv
import logging
from typing import Dict, Any, List

# Load environment variables
load_dotenv()

class LLMClient:
    def __init__(self, provider="openai", model="gpt-4o-mini", api_key=None):
        # set the provider, model and api key
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # ensure we have the api key for openai if set
        if provider == "openai" and not self.api_key:
            raise ValueError("The OPENAI_API_KEY environment variable is not set.")
        
        # check ollama is good
        if provider == "ollama" and not hasattr(ollama, "chat"):
            raise ValueError("Ollama is not properly configured in this environment.")

    def create_completion(self, messages: List[Dict], tools: List = None) -> Dict[str, Any]:
        """Create a chat completion using the specified LLM provider."""
        if self.provider == "openai":
            # perform an openai completion
            return self._openai_completion(messages, tools)
        elif self.provider == "ollama":
            # perform an ollama completion
            return self._ollama_completion(messages, tools)
        else:
            # unsupported providers
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _openai_completion(self, messages: List[Dict], tools: List) -> Dict[str, Any]:
        """Handle OpenAI chat completions."""

        # get the openai client
        client = OpenAI(api_key=self.api_key)

        try:
            # make a request, passing in tools
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or [],
            )

            # return the response
            return {
                "response": response.choices[0].message.content,
                "tool_calls": getattr(response.choices[0].message, "tool_calls", []),
            }
        except Exception as e:
            # error
            logging.error(f"OpenAI API Error: {str(e)}")
            raise ValueError(f"OpenAI API Error: {str(e)}")

    def _ollama_completion(self, messages: List[Dict], tools: List) -> Dict[str, Any]:
        """Handle Ollama chat completions."""
        # Format messages for Ollama
        ollama_messages = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in messages
        ]

        # Format tools for Ollama
        ollama_tools = []
        if tools:
            for tool in tools:
                if isinstance(tool, dict):
                    if "function" in tool:
                        # Convert OpenAI format to Ollama format
                        tool_config = {
                            "name": tool["function"]["name"],
                            "description": tool["function"].get("description", ""),
                            "parameters": tool["function"].get("parameters", {})
                        }
                        ollama_tools.append(tool_config)

        try:
            # Make API call
            response = ollama.chat(
                model="llama3.2",
                messages=ollama_messages,
                stream=False,
                tools=tools or []
            )

            logging.info(f"Ollama raw response: {response}")

            # get the message
            message = getattr(response, "message", None)

            # get the tool calls
            tool_calls = getattr(response, "tool_calls", [])

            # check for a message
            if message:
                return {"response": message.content, "tool_calls": tool_calls}
            elif hasattr(response, "error"):
                # error
                logging.error(f"Ollama API Error: {response.error}")
                raise ValueError(f"Ollama API Error: {response.error}")
            else:
                # no response
                return {"response": "No response", "tool_calls": []}

        except Exception as e:
            # error
            logging.error(f"Ollama API Error: {str(e)}")
            raise ValueError(f"Ollama API Error: {str(e)}")