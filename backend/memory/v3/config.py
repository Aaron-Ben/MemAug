"""
Graph Memory V3 — 配置
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GmConfig:
    db_path: str = "data/graph-memory/{character_id}/graph.db"
    compact_turn_count: int = 6
    recall_max_nodes: int = 6
    recall_max_depth: int = 2
    fresh_tail_count: int = 10
    dedup_threshold: float = 0.90
    pagerank_damping: float = 0.85
    pagerank_iterations: int = 20

    @classmethod
    def from_env(cls) -> "GmConfig":
        cfg = cls()
        if v := os.getenv("GM_DB_PATH"):
            cfg.db_path = v
        if v := os.getenv("GM_COMPACT_TURN_COUNT"):
            cfg.compact_turn_count = int(v)
        if v := os.getenv("GM_RECALL_MAX_NODES"):
            cfg.recall_max_nodes = int(v)
        if v := os.getenv("GM_RECALL_MAX_DEPTH"):
            cfg.recall_max_depth = int(v)
        if v := os.getenv("GM_DEDUP_THRESHOLD"):
            cfg.dedup_threshold = float(v)
        if v := os.getenv("GM_PAGERANK_DAMPING"):
            cfg.pagerank_damping = float(v)
        if v := os.getenv("GM_PAGERANK_ITERATIONS"):
            cfg.pagerank_iterations = int(v)
        return cfg
