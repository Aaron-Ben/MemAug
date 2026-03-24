"""LLM API."""

import asyncio
import json
import os
import logging
from typing import Dict, List, Optional, Union

from openai import AsyncOpenAI, OpenAI

from app.utils.json import extract_json

logger = logging.getLogger(__name__)

# LLM 单例
_llm_instance: Optional["LLM"] = None


def get_llm() -> "LLM":
    """获取 LLM 单例实例"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLM()
    return _llm_instance


class LLMConfig:
    """Configuration for LLM service."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.6,
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        base_url: Optional[str] = None,
        timeout: float = 120.0,  # HTTP timeout in seconds
        max_retries: int = 3,  # Maximum retry attempts
    ):
        """
        Initialize LLM configuration.

        Args:
            model: Model identifier (e.g., "anthropic/claude-3.5-sonnet")
            temperature: Controls randomness (0.0 to 2.0)
            api_key: OpenRouter API key
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            base_url: API base URL (defaults to OpenRouter)
            timeout: HTTP request timeout in seconds (default: 120)
            max_retries: Maximum retry attempts (default: 3)
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries


class LLM:
    """LLM service using OpenRouter API."""

    def __init__(self, config: Optional[Union[LLMConfig, Dict]] = None):
        """
        Initialize LLM service.

        Args:
            config: Configuration object or dict
        """
        if config is None:
            self.config = LLMConfig()
        elif isinstance(config, dict):
            self.config = LLMConfig(**config)
        else:
            self.config = config

        if not self.config.model:
            self.config.model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

        api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY")
        base_url = self.config.base_url or os.getenv("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable or config.api_key is required")

        # Create OpenAI client with timeout settings
        import httpx
        timeout = httpx.Timeout(self.config.timeout, connect=10.0)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=self.config.max_retries
        )
        # 异步客户端
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=self.config.max_retries
        )

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> str:
        """
        Generate a response based on the given messages.

        Args:
            messages: List of message dicts containing 'role' and 'content'.
            response_format: Optional format (e.g., {"type": "json_object"})
            tools: Optional list of tools that the model can call.
            tool_choice: Tool choice method (default: "auto").
            **kwargs: Additional parameters.

        Returns:
            The generated response.
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            **kwargs,
        }

        if response_format:
            params["response_format"] = response_format

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        logger.info(f"[LLM] Starting non-stream request to {self.config.model}")
        try:
            response = self.client.chat.completions.create(**params)
            logger.info(f"[LLM] Non-stream request completed successfully")
            return self._parse_response(response, tools)
        except Exception as e:
            logger.error(f"[LLM] Non-stream request error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a streaming response based on the given messages.

        Args:
            messages: List of message dicts containing 'role' and 'content'.
            response_format: Optional format (e.g., {"type": "json_object"})
            tools: Optional list of tools that the model can call.
            tool_choice: Tool choice method (default: "auto").
            **kwargs: Additional parameters.

        Yields:
            Chunks of the generated response.
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": True,
            **kwargs,
        }

        if response_format:
            params["response_format"] = response_format

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        try:
            stream = self.client.chat.completions.create(**params)

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield delta.content

        except Exception as e:
            logger.error(f"[LLM] API error: {type(e).__name__}: {e}")
            raise

    async def generate_response_async(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> str:
        """
        Generate a response asynchronously.

        Args:
            messages: List of message dicts containing 'role' and 'content'.
            response_format: Optional format (e.g., {"type": "json_object"})
            tools: Optional list of tools that the model can call.
            tool_choice: Tool choice method (default: "auto").
            **kwargs: Additional parameters.

        Returns:
            The generated response.
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            **kwargs,
        }

        if response_format:
            params["response_format"] = response_format

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        logger.info(f"[LLM] Starting async request to {self.config.model}")
        try:
            response = await self.async_client.chat.completions.create(**params)
            logger.info(f"[LLM] Async request completed successfully")
            return self._parse_response(response, tools)
        except Exception as e:
            logger.error(f"[LLM] Async request error: {type(e).__name__}: {e}", exc_info=True)
            raise
