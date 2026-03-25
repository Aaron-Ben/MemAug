"""
V2 Memory Backend - Session-based

基于会话的记忆系统实现
"""

import logging
from typing import Any, Dict, List, Optional

from memory.factory import MemoryBackend

logger = logging.getLogger(__name__)


class MemoryV2Backend(MemoryBackend):
    """V2 记忆系统 - 基于会话"""

    def __init__(self):
        self._chromadb_manager: Optional[Any] = None
        self._session_service: Optional[Any] = None

    @property
    def name(self) -> str:
        return "v2"

    async def initialize(self, app) -> None:
        """初始化 V2 backend"""
        # 延迟导入，避免循环依赖
        from memory.v2.chromadb_manager import ChromaDBManager
        from app.services.session_service import SessionService

        self._chromadb_manager = ChromaDBManager()
        self._session_service = SessionService(chromadb_manager=self._chromadb_manager)

        # 存储到 app state 供其他地方使用
        app.state.session_service = self._session_service
        app.state.chromadb_manager = self._chromadb_manager

        logger.info("✅ V2: SessionService initialized")

    async def search(self, query: str, character_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """通过 ChromaDB 搜索记忆"""
        if not self._chromadb_manager:
            raise RuntimeError("V2 backend not initialized")

        results = self._chromadb_manager.search_memories(
            query=query,
            character_id=character_id,
            n_results=limit
        )
        return results

    async def save_memory(self, character_id: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """保存记忆到 ChromaDB"""
        if not self._chromadb_manager:
            raise RuntimeError("V2 backend not initialized")

        memory_id = self._chromadb_manager.add_memory(
            character_id=character_id,
            content=content,
            metadata=metadata or {}
        )

        return {"status": "success", "memory_id": memory_id}

    async def get_recent_memories(self, character_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的记忆"""
        if not self._chromadb_manager:
            raise RuntimeError("V2 backend not initialized")

        # 获取最近的 session 记忆
        results = self._chromadb_manager.search_memories(
            query="",  # 空查询
            character_id=character_id,
            n_results=limit
        )
        return results


__all__ = ["MemoryV2Backend"]