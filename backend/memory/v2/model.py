from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
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
    user: str = ""     # User identifier
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
    meta: Optional[Dict[str,Any]] = None
    level: int = 2                  # 0=abstract, 1=overview, 2=detail
    vector: Optional[list] = None
    session_id: str = ""
    user: str = ""
    is_leaf: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryContext":
        """从字典创建 MemoryContext 实例"""
        return cls(
            id=data.get("id", str(uuid4())),
            uri=data.get("uri", ""),
            parent_uri=data.get("parent_uri", ""),
            category=data.get("category", ""),
            abstract=data.get("abstract", ""),
            overview=data.get("overview", ""),
            content=data.get("content", ""),
            level=data.get("level", 2),
            meta = data.get("meta", ""),
            vector=data.get("vector"),
            session_id=data.get("session_id", ""),
            user=data.get("user", ""),
            is_leaf=data.get("is_leaf", True),
        )


class DedupDecision(str, Enum):
    SKIP = "skip"
    CREATE = "create"
    NONE = "none"


class MemoryActionDecision(str, Enum):
    MERGE = "merge"
    DELETE = "delete"


@dataclass
class ExistingMemoryAction:
    """Per-memory action for deduplication (merge or delete)."""
    memory: MemoryContext
    decision: MemoryActionDecision
    reason: str


@dataclass
class DedupResult:
    """Result of deduplication decision."""
    decision: DedupDecision
    candidate: CandidateMemory
    similar_memories: List[MemoryContext]  # Similar existing memories
    actions: Optional[List[ExistingMemoryAction]] = None
    reason: str = ""
    query_vector: Optional[List[float]] = None
