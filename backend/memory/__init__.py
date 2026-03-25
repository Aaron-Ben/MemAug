"""
Memory 模块 - 记忆系统

支持两种后端：
- V1: 日记格式 (Diary-based)
- V2: 会话格式 (Session-based)

使用 MemoryBackendFactory.get_backend() 获取当前后端实例
"""

import os
import logging

logger = logging.getLogger(__name__)

# 导出工厂类和基类
from memory.factory import MemoryBackend, MemoryBackendFactory

# 兼容旧版 API（可选保留）
MEMORY_BACKEND = os.getenv("MEMORY", "v1")  # 默认 v1
V1_ENABLED = MEMORY_BACKEND == "v1"
V2_ENABLED = MEMORY_BACKEND == "v2"


def get_memory_backend() -> MemoryBackend:
    """获取当前记忆系统后端实例（兼容旧版）"""
    return MemoryBackendFactory.get_backend()


__all__ = [
    "MemoryBackend",
    "MemoryBackendFactory",
    "get_memory_backend",
    "MEMORY_BACKEND",
    "V1_ENABLED",
    "V2_ENABLED",
]
