"""Chat service v0 - Simple character chat without memory system.

Features:
- Character personality integration
- Streaming response
- No tool calling
- No memory/RAG system
"""

from typing import List, Dict, Optional, AsyncGenerator
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from app.services.llm import LLM
from app.services.character_service import CharacterStorageService
from app.models.character import UserCharacterPreference
from app.schemas.message import ChatRequest, ChatResponse, MessageContext


class ChatServiceV0:
    """
    Simple chat service without memory system.

    Features:
    - Character personality prompts
    - Streaming response
    - No tool calling
    - No RAG/memory integration
    """

    def __init__(
        self,
        llm: LLM,
        character_service: CharacterStorageService
    ):
        """
        Initialize v0 chat service.

        Args:
            llm: LLM instance to use for generating responses
            character_service: Character service for managing personalities
        """
        self.llm = llm
        self.character_service = character_service

    async def chat(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference] = None,
        user_id: str = "user_default"
    ) -> ChatResponse:
        """
        Generate a character-aware response (non-streaming).

        Args:
            request: Chat request
            user_preferences: User character preferences
            user_id: User identifier

        Returns:
            ChatResponse with message and metadata
        """
        # Collect all chunks from stream
        full_response = ""
        async for chunk in self.chat_stream(request, user_preferences, user_id):
            full_response += chunk

        # Build response object
        return ChatResponse(
            message=full_response,
            character_id=request.character_id,
            context_used=None,
            timestamp=datetime.now()
        )

    async def chat_stream(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference] = None,
        user_id: str = "user_default"
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming character-aware response.

        Args:
            request: Chat request
            user_preferences: User character preferences
            user_id: User identifier

        Yields:
            Response chunks
        """
        # Build messages
        messages = await self._build_messages(request, user_preferences, user_id)

        # Stream response from LLM
        for chunk in self.llm.generate_response_stream(messages):
            yield chunk

    async def _build_messages(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference],
        user_id: str
    ) -> List[Dict]:
        """
        Build messages list for LLM call.

        Args:
            request: Chat request
            user_preferences: User character preferences
            user_id: User identifier

        Returns:
            List of message dicts ready for LLM
        """
        # Generate system prompt from character
        system_prompt = self.character_service.get_prompt(request.character_id)
        if not system_prompt:
            raise ValueError(f"Character not found: {request.character_id}")

        # Build messages list
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if provided
        if request.conversation_history:
            messages.extend(request.conversation_history)

        # Add current message
        messages.append({"role": "user", "content": request.message})

        return messages