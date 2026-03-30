"""Chat service for V3 memory system — 知识图谱记忆

V3 特点：
1. 使用 MemoryV3Backend 进行知识图谱记忆管理
2. ingest_message 保存消息并周期性提取知识三元组
3. search 进行双路径召回（向量+FTS5+PPR）
4. 图谱上下文注入系统提示（XML 格式）
5. finalize_session 进行会话结束后的图谱维护
"""

from typing import List, Dict, Optional, AsyncGenerator
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from app.services.base_chat_service import BaseChatService
from app.services.llm import LLM
from app.services.character_service import CharacterStorageService
from app.services.chat_history_service import ChatHistoryService
from app.models.character import UserCharacterPreference
from app.schemas.message import ChatRequest, ChatResponse, MessageContext
from memory.v3.backend import MemoryV3Backend


class ChatServiceV3(BaseChatService):
    """V3 ChatService — 知识图谱驱动的记忆系统"""

    def __init__(
        self,
        llm: LLM,
        character_service: CharacterStorageService,
        history_service: ChatHistoryService,
        memory_backend: MemoryV3Backend,
    ):
        self.llm = llm
        self.character_service = character_service
        self.history_service = history_service
        self.memory_backend = memory_backend

    async def chat(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference] = None,
        user_id: str = "user_default",
    ) -> ChatResponse:
        """Generate a character-aware response with graph memory."""
        full_response = ""
        async for chunk in self.chat_stream(request, user_preferences, user_id):
            full_response += chunk

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
        user_id: str = "user_default",
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming character-aware response with graph memory."""
        messages = await self._build_messages(request, user_preferences, user_id)

        for chunk in self.llm.generate_response_stream(messages):
            yield chunk

    async def _build_messages(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference],
        user_id: str,
    ) -> List[Dict]:
        """构建消息列表：角色提示 + 图谱记忆 + 对话历史 + 当前消息"""
        # 1. 角色系统提示
        system_prompt = self.character_service.get_prompt(request.character_id)
        if not system_prompt:
            raise ValueError(f"Character not found: {request.character_id}")

        # 2. 召回图谱记忆并注入系统提示
        try:
            search_results = await self.memory_backend.search(
                query=request.message,
                character_id=request.character_id,
            )
            if search_results:
                result = search_results[0]
                graph_prompt = result.get("system_prompt", "")
                graph_xml = result.get("xml", "")
                episodic_xml = result.get("episodic_xml", "")

                if graph_xml:
                    parts = [system_prompt]
                    if graph_prompt:
                        parts.append(graph_prompt)
                    parts.append(graph_xml)
                    if episodic_xml:
                        parts.append(episodic_xml)
                    system_prompt = "\n\n".join(parts)

                    logger.info(
                        f"[V3] Injected graph memory: "
                        f"{len(result.get('nodes', []))} nodes, "
                        f"{len(result.get('edges', []))} edges, "
                        f"~{result.get('tokens', 0)} tokens"
                    )
        except Exception as e:
            logger.warning(f"[V3] Graph memory recall failed: {e}")

        # 3. 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]

        # 4. 对话历史
        if request.conversation_history:
            messages.extend(request.conversation_history)

        # 5. 当前消息
        messages.append({"role": "user", "content": request.message})

        return messages

    async def persist_messages(
        self,
        character_id: str,
        topic_id: int,
        user_id: str,
        character_name: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """双写：history_service（对话连续性）+ ingest_message（知识提取）"""
        # 1. 保存到 history_service
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

        # 2. 写入图谱（触发周期性知识提取，内部 asyncio.create_task）
        session_id = str(topic_id)
        try:
            await self.memory_backend.ingest_message(
                character_id=character_id,
                session_id=session_id,
                role="user",
                content=user_message,
            )
            await self.memory_backend.ingest_message(
                character_id=character_id,
                session_id=session_id,
                role="assistant",
                content=assistant_message,
            )
            logger.info(f"[V3] Ingested messages for session={session_id}")
        except Exception as e:
            logger.error(f"[V3] Failed to ingest messages: {e}")

    async def finalize_session(
        self,
        character_id: str,
        session_id: str,
    ) -> Dict:
        """结束会话：EVENT→PATTERN 升级 + 图谱维护"""
        return await self.memory_backend.finalize_session(character_id, session_id)
