"""Chat service for V2 memory system.

V2 特点：
1. 使用 HierarchicalRetriever 进行层级记忆检索
2. 使用 SessionService 管理会话（自动 commit）
3. 不使用 plugin_manager（tool calling）
4. 集成日记生成和记忆压缩
"""

from typing import List, Dict, Optional, AsyncGenerator, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from app.services.llm import LLM
from app.services.character_service import CharacterStorageService
from app.models.character import UserCharacterPreference
from app.schemas.message import (
    ChatRequest,
    ChatResponse,
    MessageContext
)
from memory.v2.backend import MemoryBackend
from memory.v2.retriever import SpaceType


class ChatServiceV2:
    """
    Chat service for V2 memory system.

    特性：
    - 层级记忆检索 (HierarchicalRetriever)
    - 会话自动提交 (SessionService)
    - 支持日记生成和记忆压缩
    """

    def __init__(
        self,
        llm: LLM,
        character_service: CharacterStorageService,
        memory_backend: Optional[MemoryBackend] = None
    ):
        """
        Initialize chat service V2.

        Args:
            llm: LLM instance to use for generating responses
            character_service: Character service for managing personalities
            memory_backend: V2 memory backend (optional, for direct memory access)
        """
        self.llm = llm
        self.character_service = character_service
        self.memory_backend = memory_backend
        self.max_tool_iterations = 0  # V2 不使用 tool calling

    def _build_message_context(
        self,
        request: ChatRequest
    ) -> Optional[MessageContext]:
        """Build message context based on request metadata."""
        character = self.character_service.get_character(request.character_id)
        if not character:
            return None

        # Default behavior parameters
        character_state = {"proactivity_level": 0.5, "argument_avoidance_threshold": 0.7}
        initiate_topic = False  # V2 不主动发起话题

        return MessageContext(
            character_state=character_state,
            initiate_topic=initiate_topic
        )

    async def chat(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference] = None,
        user_id: str = "user_default",
        memory_context: str = "",
        session_service=None,
        retriever=None
    ) -> ChatResponse:
        """
        Generate a character-aware response with V2 memory integration.
        """
        # Collect all chunks from stream
        full_response = ""
        async for chunk in self.chat_stream(
            request, user_preferences, user_id, memory_context, session_service, retriever
        ):
            full_response += chunk

        # Build response object
        message_context = self._build_message_context(request)
        return ChatResponse(
            message=full_response,
            character_id=request.character_id,
            context_used=message_context.dict() if message_context else None,
            timestamp=datetime.now()
        )

    async def chat_stream(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference] = None,
        user_id: str = "user_default",
        memory_context: str = "",
        session_service=None,
        retriever=None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming character-aware response.

        V2 流程：
        1. 构建消息（含记忆上下文）
        2. 流式生成回复
        3. 不使用 tool calling
        """
        # Build initial messages
        messages = await self._build_messages(
            request, user_preferences, user_id, memory_context
        )

        # Stream response (no tool calling in V2)
        for chunk in self.llm.generate_response_stream(messages):
            yield chunk

    async def _build_messages(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference],
        user_id: str,
        memory_context: str = ""
    ) -> List[Dict]:
        """
        Build messages list for LLM call.

        Returns:
            List of message dicts ready for LLM
        """
        # Generate system prompt
        system_prompt = self.character_service.get_prompt(request.character_id)
        if not system_prompt:
            raise ValueError(f"Character not found: {request.character_id}")

        # Build messages list
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if provided
        if request.conversation_history:
            messages.extend(request.conversation_history)

        # Add memory context before current message (from retriever)
        if memory_context:
            messages.append({"role": "user", "content": memory_context})

        # Add current message
        messages.append({"role": "user", "content": request.message})

        return messages

    async def retrieve_memory(
        self,
        query: str,
        user_id: str,
        retriever,
        limit: int = 5
    ) -> str:
        """
        Retrieve relevant memories for the query.

        Args:
            query: User's message
            user_id: User identifier
            retriever: HierarchicalRetriever instance
            limit: Maximum number of contexts to retrieve

        Returns:
            Formatted memory context string
        """
        if not retriever:
            return ""

        try:
            result = await retriever.retrieve(
                query=query,
                user=user_id,
                space=SpaceType.USER,
                limit=limit
            )

            if result.matched_contexts:
                memory_parts = ["[相关记忆参考]"]
                for ctx in result.matched_contexts:
                    # 截断过长内容
                    content = ctx.abstract[:300] if len(ctx.abstract) > 300 else ctx.abstract
                    memory_parts.append(f"- {content}")
                return "\n".join(memory_parts)
        except Exception as e:
            logger.warning(f"[Memory] Failed to retrieve: {e}")

        return ""

    async def save_to_memory(
        self,
        session_service,
        character_id: str,
        topic_id: str,
        user_id: str,
        character_name: str,
        user_message: str,
        assistant_message: str
    ):
        """
        Save conversation to V2 memory system.

        Args:
            session_service: SessionService instance
            character_id: Character identifier
            topic_id: Topic identifier
            user_id: User identifier
            character_name: Character name
            user_message: User's message
            assistant_message: Assistant's response
        """
        if not session_service:
            return

        try:
            await session_service.add_message(
                character_id=character_id,
                topic_id=topic_id,
                role="user",
                content=user_message,
                name=user_id,
                user_id=user_id
            )
            await session_service.add_message(
                character_id=character_id,
                topic_id=topic_id,
                role="assistant",
                content=assistant_message,
                name=character_name,
                user_id=user_id
            )
            logger.info(f"[Memory] Saved conversation to session")
        except Exception as e:
            logger.error(f"[Memory] Failed to save: {e}")