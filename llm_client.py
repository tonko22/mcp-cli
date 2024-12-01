import os
import ollama
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMClient:
    def __init__(self, provider="openai", model="gpt-4o-mini", api_key=None):
        """
        Initialize the LLM client.

        :param provider: The provider of the LLM, either 'openai' or 'ollama'.
        :param model: The model to use.
        :param api_key: API key for OpenAI (if using OpenAI).
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if provider == "openai" and not self.api_key:
            raise ValueError("The OPENAI_API_KEY environment variable is not set.")

        if provider == "ollama" and not hasattr(ollama, "chat"):
            raise ValueError("Ollama is not properly configured in this environment.")

    def create_completion(self, messages, tools=None):
        """
        Create a chat completion using the specified LLM provider.

        :param messages: Conversation history as a list of dictionaries.
        :param tools: List of tools (applicable for both OpenAI and Ollama).
        :return: A normalized response object.
        """
        if self.provider == "openai":
            return self._openai_completion(messages, tools)
        elif self.provider == "ollama":
            return self._ollama_completion(messages, tools)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _openai_completion(self, messages, tools):
        """Handle OpenAI chat completions."""
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or [],
        )
        # Normalize OpenAI response for consistency
        return {
            "response": response.choices[0].message.content,
            "tool_calls": getattr(response.choices[0].message, "tool_calls", []),
        }

    def _ollama_completion(self, messages, tools):
        """Handle Ollama chat completions."""
        ollama_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]
        response = ollama.chat(
            "llama3.1",
            messages=ollama_messages,
            tools=tools,  # Pass tools to enable tool support
        )

        # Debugging: Log the raw Ollama response
        print(f"Ollama raw response: {response}")

        # Normalize Ollama response for consistency
        if hasattr(response, "message") and response.message:
            tool_calls = getattr(response, "tool_calls", [])
            return {"response": response.message.content, "tool_calls": tool_calls}
        elif hasattr(response, "error"):
            raise ValueError(f"Ollama API Error: {response.error}")
        else:
            return {"response": "No response", "tool_calls": []}
