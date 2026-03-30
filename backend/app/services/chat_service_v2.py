"""Chat service for V2 memory system.

V2 特点：
1. 使用 HierarchicalRetriever 进行层级记忆检索
2. 使用 SessionService 管理会话（自动 commit）
3. 不使用 plugin_manager（tool calling）
4. 集成日记生成和记忆压缩
5. 集成 Skills 系统（通过 system prompt 注入）
"""

from typing import List, Dict, Optional, AsyncGenerator, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from app.services.base_chat_service import BaseChatService
from app.services.llm import LLM
from app.services.character_service import CharacterStorageService
from app.services.chat_history_service import ChatHistoryService
from app.services.embedding import EmbeddingService
from app.models.character import UserCharacterPreference
from app.schemas.message import (
    ChatRequest,
    ChatResponse,
    MessageContext
)
from app.skills.loader import get_skills_loader
from memory.v2.backend import MemoryBackend
from memory.v2.chromadb_manager import ChromaDBManager
from memory.v2.retriever import HierarchicalRetriever, SpaceType
from app.services.session_service import SessionService


class ChatServiceV2(BaseChatService):
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
        history_service: ChatHistoryService,
        memory_backend: Optional[MemoryBackend] = None,
    ):
        self.llm = llm
        self.character_service = character_service
        self.history_service = history_service
        self.memory_backend = memory_backend
        self.max_tool_iterations = 0  # V2 不使用 tool calling

        # 初始化 V2 专属服务
        self._chromadb_manager = ChromaDBManager()
        self._session_service = SessionService(chromadb_manager=self._chromadb_manager)
        self._embedding_service = EmbeddingService()
        self._retriever = HierarchicalRetriever(
            chromadb_manager=self._chromadb_manager,
            embedding_service=self._embedding_service,
        )

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
    ) -> ChatResponse:
        """Generate a character-aware response with V2 memory integration."""
        # Collect all chunks from stream
        full_response = ""
        async for chunk in self.chat_stream(request, user_preferences, user_id):
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
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming character-aware response."""
        # Build initial messages (内含记忆检索)
        messages = await self._build_messages(request, user_preferences, user_id)

        # Stream response (no tool calling in V2)
        for chunk in self.llm.generate_response_stream(messages):
            yield chunk

    async def _build_messages(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference],
        user_id: str,
    ) -> List[Dict]:
        """Build messages list for LLM call."""
        # Generate character prompt
        character_prompt = self.character_service.get_prompt(request.character_id)
        if not character_prompt:
            raise ValueError(f"Character not found: {request.character_id}")

        # Build skills content
        skills_loader = get_skills_loader()
        # 加载所有可用 skill（不依赖 always 标志）
        all_skills = [s["name"] for s in skills_loader.list_skills() if s["available"]]
        always_content = skills_loader.load_skills_for_context(all_skills) if all_skills else ""
        skills_summary = skills_loader.build_skills_summary()

        # Combine character prompt with skills
        parts = [character_prompt]

        # Add always-loaded skills (Active Skills)
        if always_content:
            parts.append(f"# Active Skills\n\n{always_content}")

        # Add skills summary (Available Skills)
        if skills_summary:
            parts.append(f"""# Skills
The following skills extend your capabilities. You can read their SKILL.md files for details.

{skills_summary}""")

        system_prompt = "\n\n".join(parts)

        # Build messages list
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if provided
        if request.conversation_history:
            messages.extend(request.conversation_history)

        # 记忆检索（内化，不再由路由层传入）
        memory_context = await self._retrieve_memory(request.message, user_id)
        if memory_context:
            messages.append({"role": "user", "content": memory_context})

        # Add current message
        messages.append({"role": "user", "content": request.message})

        return messages

    async def _retrieve_memory(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> str:
        """检索相关记忆"""
        if not self._retriever:
            return ""

        try:
            result = await self._retriever.retrieve(
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

    async def persist_messages(
        self,
        character_id: str,
        topic_id: int,
        user_id: str,
        character_name: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """保存对话消息：history_service + session_service 双写"""
        # 1. 保存到 history_service（对话连续性）
        self.history_service.append_message(
            user_id=user_id, topic_id=topic_id,
            role="user", content=user_message,
            name=user_id, character_id=character_id,
        )
        self.history_service.append_message(
            user_id=user_id, topic_id=topic_id,
            role="assistant", content=assistant_message,
            name=character_name, character_id=character_id,
        )

        # 2. 保存到 V2 session service
        try:
            await self._session_service.add_message(
                character_id=character_id,
                topic_id=topic_id,
                role="user",
                content=user_message,
                name=user_id,
                user_id=user_id
            )
            await self._session_service.add_message(
                character_id=character_id,
                topic_id=topic_id,
                role="assistant",
                content=assistant_message,
                name=character_name,
                user_id=user_id
            )
            logger.info(f"[Memory] Saved conversation to session")
        except Exception as e:
            logger.error(f"[Memory] Failed to save to session: {e}")
