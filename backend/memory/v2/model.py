from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime
from uuid import uuid4


class MemoryCategory(str, Enum):
    PROFILE = "profile"           # 用户基本信息
    PREFERENCES = "preferences"   # 用户偏好
    ENTITIES = "entities"         # 实体记忆（人、项目）
    EVENTS = "events"             # 事件记录
    CASES = "cases"               # Agent 学习到的案例
    PATTERNS = "patterns"         # Agent 学习到的模式


@dataclass
class CandidateMemory:
    """从会话中提取的候选记忆"""
    category: MemoryCategory
    abstract: str      # L0: ~100 tokens
    overview: str      # L1: ~500 tokens
    content: str       # L2: 完整内容
    source_session: str
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class MergedMemoryPayload:
    """Structured merged memory payload returned by one LLM call."""

    abstract: str
    overview: str
    content: str
    reason: str = ""

@dataclass
class MemoryContext:
    """存储到向量数据库的记忆上下文"""
    id: str = field(default_factory=lambda: str(uuid4()))
    uri: str = ""                    # 如: memories/entities/mem_xxx.md
    parent_uri: str = ""
    category: str = ""
    abstract: str = ""               # L0
    overview: str = ""               # L1
    content: str = ""                # L2 (向量化内容)
    level: int = 2                  # 0=abstract, 1=overview, 2=detail
    vector: Optional[list] = None
    session_id: str = ""
    user: str = ""
    is_leaf: bool = True