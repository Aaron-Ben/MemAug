"""
Graph Memory V3 — 向量余弦去重

发现并合并语义重复的节点
"""

from __future__ import annotations

import logging
import math
import sqlite3
from typing import Any, Dict, List

from ..config import GmConfig
from ..types import DedupResult
from ..store.store import find_by_id, merge_nodes, get_all_vectors

logger = logging.getLogger(__name__)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-9)


def detect_duplicates(db: sqlite3.Connection, cfg: GmConfig) -> List[Dict[str, Any]]:
    vectors = get_all_vectors(db)
    if len(vectors) < 2:
        return []

    threshold = cfg.dedup_threshold
    pairs = []

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = _cosine_sim(vectors[i]["embedding"], vectors[j]["embedding"])
            if sim >= threshold:
                node_a = find_by_id(db, vectors[i]["nodeId"])
                node_b = find_by_id(db, vectors[j]["nodeId"])
                if node_a and node_b:
                    pairs.append({
                        "nodeA": node_a.id,
                        "nodeB": node_b.id,
                        "nameA": node_a.name,
                        "nameB": node_b.name,
                        "similarity": sim,
                    })

    pairs.sort(key=lambda p: p["similarity"], reverse=True)
    return pairs


def dedup(db: sqlite3.Connection, cfg: GmConfig) -> DedupResult:
    pairs = detect_duplicates(db, cfg)
    merged_count = 0
    consumed: set = set()

    for pair in pairs:
        if pair["nodeA"] in consumed or pair["nodeB"] in consumed:
            continue

        a = find_by_id(db, pair["nodeA"])
        b = find_by_id(db, pair["nodeB"])
        if not a or not b:
            continue

        # 只合并同类型
        if a.type != b.type:
            continue

        # 决定保留哪个
        if a.validated_count > b.validated_count:
            keep_id, merge_id = a.id, b.id
        elif b.validated_count > a.validated_count:
            keep_id, merge_id = b.id, a.id
        else:
            if a.updated_at >= b.updated_at:
                keep_id, merge_id = a.id, b.id
            else:
                keep_id, merge_id = b.id, a.id

        merge_nodes(db, keep_id, merge_id)
        consumed.add(merge_id)
        merged_count += 1

    return DedupResult(pairs=pairs, merged=merged_count)
