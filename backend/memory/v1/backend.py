"""
V1 Memory Backend - Diary-based

基于日记的记忆系统实现
"""

import logging
from typing import Any, Dict, List, Optional

from memory.factory import MemoryBackend
from memory.v1.services.diary import DiaryFileService
from memory.v1.plugin_manager import plugin_manager

logger = logging.getLogger(__name__)


class MemoryV1Backend(MemoryBackend):
    """V1 记忆系统 - 基于日记"""

    @property
    def name(self) -> str:
        return "v1"

    async def initialize(self, app) -> None:
        """初始化 V1 backend"""
        from app.vector_index import initialize_vector_index

        # 初始化向量索引
        vector_index = await initialize_vector_index()
        plugin_manager.set_vector_db_manager(vector_index)
        logger.info("✅ V1: VectorIndex initialized")

        # 加载插件
        await plugin_manager.load_plugins()
        logger.info(f"✅ V1: Plugins loaded: {list(plugin_manager.plugins.keys())}")

        # 启动时同步日记到向量索引
        from app.vector_index import sync_all_diaries_to_vector_index
        await sync_all_diaries_to_vector_index()
        logger.info("✅ V1: Diary sync completed")

    async def search(self, query: str, character_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """通过向量索引搜索日记"""
        from app.vector_index import search_diaries

        results = await search_diaries(
            query=query,
            character_id=character_id,
            limit=limit
        )
        return results

    async def save_memory(self, character_id: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """V1 模式下通过日记插件保存"""
        # 调用 DailyNote 插件创建日记
        from datetime import datetime

        date = metadata.get("date", datetime.now().strftime("%Y-%m-%d")) if metadata else datetime.now().strftime("%Y-%m-%d")
        tag = metadata.get("tag") if metadata else None

        result = await plugin_manager.process_tool_call("DailyNote", {
            "command": "create",
            "maid": character_id,
            "date": date,
            "content": content,
            "tag": tag
        })

        return result

    async def get_recent_memories(self, character_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的日记"""
        diary_service = DiaryFileService()
        diaries = diary_service.list_diaries(character_id=character_id, limit=limit)
        return diaries


__all__ = ["MemoryV1Backend"]