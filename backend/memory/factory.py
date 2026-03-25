"""
Memory Backend 抽象层

提供统一的接口用于切换 v1 (Diary-based) 和 v2 (Session-based) 记忆系统
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryBackend(ABC):
    """记忆系统抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """返回 backend 名称 (v1/v2)"""
        pass

    @abstractmethod
    async def initialize(self, app) -> None:
        """初始化 backend，在应用启动时调用"""
        pass

    @abstractmethod
    async def search(self, query: str, character_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索记忆"""
        pass

    @abstractmethod
    async def save_memory(self, character_id: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """保存记忆"""
        pass

    @abstractmethod
    async def get_recent_memories(self, character_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的记忆"""
        pass


class MemoryBackendFactory:
    """记忆系统工厂类"""

    _backends: Dict[str, type] = {}
    _instance: Optional[MemoryBackend] = None
    _registered: bool = False

    @classmethod
    def register(cls, name: str, backend_class: type) -> None:
        """注册一个 backend 实现"""
        cls._backends[name] = backend_class
        logger.info(f"[MemoryBackendFactory] Registered: {name}")

    @classmethod
    def _ensure_registered(cls) -> None:
        """确保 backend 已注册（延迟注册）"""
        if cls._registered:
            return
        from memory.v1 import MemoryV1Backend
        from memory.v2 import MemoryV2Backend
        cls.register("v1", MemoryV1Backend)
        cls.register("v2", MemoryV2Backend)
        cls._registered = True

    @classmethod
    def get_backend(cls, name: Optional[str] = None) -> MemoryBackend:
        """获取 backend 实例（单例）"""
        if cls._instance is not None:
            return cls._instance

        cls._ensure_registered()

        name = name or os.getenv("MEMORY", "v1")

        if name not in cls._backends:
            raise ValueError(f"Unknown memory backend: {name}. Available: {list(cls._backends.keys())}")

        cls._instance = cls._backends[name]()
        logger.info(f"[MemoryBackendFactory] Created instance: {name}")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置实例（用于测试）"""
        cls._instance = None

    @classmethod
    def get_current_backend_name(cls) -> str:
        """获取当前 backend 名称"""
        return os.getenv("MEMORY", "v1")


__all__ = ["MemoryBackend", "MemoryBackendFactory"]