"""
Graph Memory V3 — Personalized PageRank (PPR) + 全局 PageRank

个性化 PPR：从种子节点出发传播权重，离种子越近分数越高
全局 PageRank：均匀 teleport，写入 gm_nodes.pagerank 作为基线
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Dict, List, Optional, Set

from ..config import GmConfig
from ..types import GlobalPageRankResult
from ..store.store import update_pageranks

logger = logging.getLogger(__name__)

# ─── 图结构缓存 ─────────────────────────────────────────────


class _GraphStructure:
    __slots__ = ("node_ids", "adj", "N", "cached_at")

    def __init__(self):
        self.node_ids: Set[str] = set()
        self.adj: Dict[str, List[str]] = {}
        self.N: int = 0
        self.cached_at: float = 0


_cached: Optional[_GraphStructure] = None
_CACHE_TTL = 30.0  # 30 秒


def _load_graph(db: sqlite3.Connection) -> _GraphStructure:
    global _cached
    if _cached and time.time() - _cached.cached_at < _CACHE_TTL:
        return _cached

    node_rows = db.execute("SELECT id FROM gm_nodes WHERE status='active'").fetchall()
    node_ids = {r[0] for r in node_rows}

    edge_rows = db.execute("SELECT from_id, to_id FROM gm_edges").fetchall()
    adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}

    for e in edge_rows:
        fid, tid = e[0], e[1]
        if fid in node_ids and tid in node_ids:
            adj[fid].append(tid)
            adj[tid].append(fid)

    gs = _GraphStructure()
    gs.node_ids = node_ids
    gs.adj = adj
    gs.N = len(node_ids)
    gs.cached_at = time.time()
    _cached = gs
    return gs


def invalidate_graph_cache() -> None:
    global _cached
    _cached = None


# ─── 个性化 PageRank ────────────────────────────────────────


def personalized_page_rank(
    db: sqlite3.Connection,
    seed_ids: List[str],
    candidate_ids: List[str],
    cfg: GmConfig,
) -> Dict[str, float]:
    graph = _load_graph(db)
    damping = cfg.pagerank_damping
    iterations = cfg.pagerank_iterations

    if graph.N == 0 or not seed_ids:
        return {}

    valid_seeds = [sid for sid in seed_ids if sid in graph.node_ids]
    if not valid_seeds:
        return {}

    teleport_weight = 1.0 / len(valid_seeds)
    seed_set = set(valid_seeds)

    # 初始分数集中在种子节点
    rank = {nid: (teleport_weight if nid in seed_set else 0.0) for nid in graph.node_ids}

    for _ in range(iterations):
        new_rank: Dict[str, float] = {}

        # teleport 分量
        for nid in graph.node_ids:
            new_rank[nid] = (1 - damping) * teleport_weight if nid in seed_set else 0.0

        # 传播分量
        for nid, neighbors in graph.adj.items():
            if not neighbors:
                continue
            contrib = rank.get(nid, 0.0) / len(neighbors)
            if contrib == 0:
                continue
            for nb in neighbors:
                new_rank[nb] = new_rank.get(nb, 0.0) + damping * contrib

        # dangling nodes 传播回种子
        dangling_sum = sum(
            rank.get(nid, 0.0)
            for nid in graph.node_ids
            if not graph.adj.get(nid)
        )
        if dangling_sum > 0:
            dc = damping * dangling_sum * teleport_weight
            for sid in valid_seeds:
                new_rank[sid] = new_rank.get(sid, 0.0) + dc

        rank = new_rank

    return {cid: rank.get(cid, 0.0) for cid in candidate_ids}


# ─── 全局 PageRank ──────────────────────────────────────────


def compute_global_page_rank(
    db: sqlite3.Connection,
    cfg: GmConfig,
) -> GlobalPageRankResult:
    graph = _load_graph(db)
    damping = cfg.pagerank_damping
    iterations = cfg.pagerank_iterations

    if graph.N == 0:
        return GlobalPageRankResult(scores={}, top_k=[])

    name_rows = db.execute("SELECT id, name FROM gm_nodes WHERE status='active'").fetchall()
    name_map = {r[0]: r[1] for r in name_rows}

    init = 1.0 / graph.N
    rank = {nid: init for nid in graph.node_ids}

    for _ in range(iterations):
        base = (1 - damping) / graph.N
        new_rank = {nid: base for nid in graph.node_ids}

        for nid, neighbors in graph.adj.items():
            if not neighbors:
                continue
            contrib = rank.get(nid, 0.0) / len(neighbors)
            for nb in neighbors:
                new_rank[nb] = new_rank.get(nb, base) + damping * contrib

        dangling_sum = sum(
            rank.get(nid, 0.0)
            for nid in graph.node_ids
            if not graph.adj.get(nid)
        )
        if dangling_sum > 0:
            dc = damping * dangling_sum / graph.N
            for nid in graph.node_ids:
                new_rank[nid] = new_rank.get(nid, 0.0) + dc

        rank = new_rank

    update_pageranks(db, rank)

    sorted_scores = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:20]
    top_k = [{"id": id_, "name": name_map.get(id_, id_), "score": score} for id_, score in sorted_scores]

    return GlobalPageRankResult(scores=rank, top_k=top_k)
