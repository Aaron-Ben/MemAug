"""
Graph Memory V3 — 类型定义

节点：USER / PERSON / TOPIC / EVENT / PATTERN / CASE / PREFERENCE
边：CARES_ABOUT / INVOLVED_IN / TRIGGERS / LEADS_TO / HAS_PREFERENCE / RESOLVED_BY / RELATED_TO
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── 节点 ─────────────────────────────────────────────────────


class NodeType(str, enum.Enum):
    USER = "USER"
    PERSON = "PERSON"
    TOPIC = "TOPIC"
    EVENT = "EVENT"
    PATTERN = "PATTERN"
    CASE = "CASE"
    PREFERENCE = "PREFERENCE"


class NodeStatus(str, enum.Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclass
class GmNode:
    id: str
    type: NodeType
    name: str
    description: str
    content: str
    status: NodeStatus
    validated_count: int
    source_sessions: List[str]
    community_id: Optional[str]
    pagerank: float
    created_at: int
    updated_at: int


# ─── 边 ───────────────────────────────────────────────────────


class EdgeType(str, enum.Enum):
    CARES_ABOUT = "CARES_ABOUT"
    INVOLVED_IN = "INVOLVED_IN"
    TRIGGERS = "TRIGGERS"
    LEADS_TO = "LEADS_TO"
    HAS_PREFERENCE = "HAS_PREFERENCE"
    RESOLVED_BY = "RESOLVED_BY"
    RELATED_TO = "RELATED_TO"


@dataclass
class GmEdge:
    id: str
    from_id: str
    to_id: str
    type: EdgeType
    instruction: str
    condition: Optional[str]
    session_id: str
    created_at: int


# ─── 信号 ─────────────────────────────────────────────────────


class SignalType(str, enum.Enum):
    TOOL_ERROR = "tool_error"
    TOOL_SUCCESS = "tool_success"
    SKILL_INVOKED = "skill_invoked"
    USER_CORRECTION = "user_correction"
    EXPLICIT_RECORD = "explicit_record"
    TASK_COMPLETED = "task_completed"


@dataclass
class Signal:
    type: SignalType
    turn_index: int
    data: Dict[str, Any]


# ─── 提取结果 ─────────────────────────────────────────────────


@dataclass
class NodeCandidate:
    type: NodeType
    name: str
    description: str
    content: str


@dataclass
class EdgeCandidate:
    from_name: str
    to_name: str
    type: EdgeType
    instruction: str
    condition: Optional[str] = None


@dataclass
class ExtractionResult:
    nodes: List[NodeCandidate]
    edges: List[EdgeCandidate]


@dataclass
class PromotedPattern:
    name: str
    description: str
    content: str


@dataclass
class FinalizeResult:
    promoted_patterns: List[PromotedPattern]
    new_edges: List[EdgeCandidate]
    invalidations: List[str]


# ─── 召回结果 ─────────────────────────────────────────────────


@dataclass
class RecallResult:
    nodes: List[GmNode]
    edges: List[GmEdge]
    token_estimate: int


# ─── 向量搜索结果 ─────────────────────────────────────────────


@dataclass
class ScoredNode:
    node: GmNode
    score: float


@dataclass
class ScoredCommunity:
    id: str
    summary: str
    score: float
    node_count: int


# ─── 社区 ─────────────────────────────────────────────────────


@dataclass
class CommunitySummary:
    id: str
    summary: str
    node_count: int
    created_at: int
    updated_at: int


# ─── 维护结果 ─────────────────────────────────────────────────


@dataclass
class DedupResult:
    pairs: List[Dict[str, Any]]
    merged: int


@dataclass
class GlobalPageRankResult:
    scores: Dict[str, float]
    top_k: List[Dict[str, Any]]


@dataclass
class CommunityResult:
    labels: Dict[str, str]
    communities: Dict[str, List[str]]
    count: int


@dataclass
class MaintenanceResult:
    dedup: DedupResult
    pagerank: GlobalPageRankResult
    community: CommunityResult
    community_summaries: int
    duration_ms: int
