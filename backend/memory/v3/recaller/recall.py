"""
Graph Memory V3 — 双路径召回引擎

精确路径（向量/FTS5 → 社区扩展 → 图遍历 → PPR 排序）
泛化路径（社区向量搜索 → 匹配社区成员 → 图遍历 → PPR 排序）
合并策略：精确优先，泛化补充
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from typing import Any, Callable, Coroutine, Dict, List, Optional

from ..config import GmConfig
from ..types import GmEdge, GmNode, RecallResult
from ..store.store import (
    search_nodes,
    vector_search_with_score,
    graph_walk,
    community_representatives,
    community_vector_search,
    nodes_by_community_ids,
    save_vector,
    get_vector_hash,
)
from ..graph.pagerank import personalized_page_rank
from ..graph.community import get_community_peers

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Coroutine[Any, Any, List[float]]]


class Recaller:
    def __init__(self, db: sqlite3.Connection, cfg: GmConfig):
        self._db = db
        self._cfg = cfg
        self._embed: Optional[EmbedFn] = None

    def set_embed_fn(self, fn: EmbedFn) -> None:
        self._embed = fn

    async def recall(self, query: str) -> RecallResult:
        limit = self._cfg.recall_max_nodes

        precise = await self._recall_precise(query, limit)
        generalized = await self._recall_generalized(query, limit)
        merged = self._merge_results(precise, generalized)

        if os.environ.get("GM_DEBUG"):
            communities = {n.community_id for n in merged.nodes if n.community_id}
            logger.debug(
                f"recall merged: precise={len(precise.nodes)}, generalized={len(generalized.nodes)} "
                f"→ final={len(merged.nodes)} nodes, {len(merged.edges)} edges, {len(communities)} communities"
            )

        return merged

    async def _recall_precise(self, query: str, limit: int) -> RecallResult:
        seeds: List[GmNode] = []

        if self._embed:
            try:
                vec = await self._embed(query)
                scored = vector_search_with_score(self._db, vec, max(limit // 2, 1))
                seeds = [s.node for s in scored]

                if os.environ.get("GM_DEBUG") and scored:
                    logger.debug(f"precise: bestScore={scored[0].score:.3f}, seeds={len(seeds)}")

                if len(seeds) < 2:
                    fts = search_nodes(self._db, query, limit)
                    seen = {n.id for n in seeds}
                    seeds.extend(n for n in fts if n.id not in seen)
            except Exception:
                seeds = search_nodes(self._db, query, limit)
        else:
            seeds = search_nodes(self._db, query, limit)

        if not seeds:
            return RecallResult(nodes=[], edges=[], token_estimate=0)

        seed_ids = [n.id for n in seeds]

        # 社区扩展
        expanded_ids = set(seed_ids)
        for seed in seeds:
            peers = get_community_peers(self._db, seed.id, 2)
            expanded_ids.update(peers)

        nodes, edges = graph_walk(self._db, list(expanded_ids), self._cfg.recall_max_depth)
        if not nodes:
            return RecallResult(nodes=[], edges=[], token_estimate=0)

        candidate_ids = [n.id for n in nodes]
        ppr_scores = personalized_page_rank(self._db, seed_ids, candidate_ids, self._cfg)

        filtered = sorted(
            nodes,
            key=lambda n: (
                -(ppr_scores.get(n.id, 0)),
                -n.validated_count,
                -n.updated_at,
            ),
        )[:limit]

        ids = {n.id for n in filtered}
        return RecallResult(
            nodes=filtered,
            edges=[e for e in edges if e.from_id in ids and e.to_id in ids],
            token_estimate=self._estimate_tokens(filtered),
        )

    async def _recall_generalized(self, query: str, limit: int) -> RecallResult:
        seeds: List[GmNode] = []

        if self._embed:
            try:
                vec = await self._embed(query)
                scored_communities = community_vector_search(self._db, vec)
                if scored_communities:
                    community_ids = [c.id for c in scored_communities]
                    seeds = nodes_by_community_ids(self._db, community_ids, 3)

                    if os.environ.get("GM_DEBUG"):
                        logger.debug(
                            f"generalized: community vector matched {len(scored_communities)} communities: "
                            + ", ".join(f"{c.id}({c.score:.2f})" for c in scored_communities)
                        )
            except Exception:
                pass

        if not seeds:
            seeds = community_representatives(self._db, 2)

        if not seeds:
            return RecallResult(nodes=[], edges=[], token_estimate=0)

        seed_ids = [n.id for n in seeds]
        nodes, edges = graph_walk(self._db, seed_ids, 1)
        if not nodes:
            return RecallResult(nodes=[], edges=[], token_estimate=0)

        candidate_ids = [n.id for n in nodes]
        ppr_scores = personalized_page_rank(self._db, seed_ids, candidate_ids, self._cfg)

        filtered = sorted(
            nodes,
            key=lambda n: (
                -(ppr_scores.get(n.id, 0)),
                -n.updated_at,
                -n.validated_count,
            ),
        )[:limit]

        ids = {n.id for n in filtered}
        return RecallResult(
            nodes=filtered,
            edges=[e for e in edges if e.from_id in ids and e.to_id in ids],
            token_estimate=self._estimate_tokens(filtered),
        )

    def _merge_results(self, precise: RecallResult, generalized: RecallResult) -> RecallResult:
        node_map: Dict[str, GmNode] = {}
        edge_map: Dict[str, GmEdge] = {}

        for n in precise.nodes:
            node_map[n.id] = n
        for e in precise.edges:
            edge_map[e.id] = e

        for n in generalized.nodes:
            if n.id not in node_map:
                node_map[n.id] = n

        final_ids = set(node_map.keys())
        for e in generalized.edges:
            if e.id not in edge_map and e.from_id in final_ids and e.to_id in final_ids:
                edge_map[e.id] = e

        nodes = list(node_map.values())
        edges = list(edge_map.values())
        return RecallResult(nodes=nodes, edges=edges, token_estimate=self._estimate_tokens(nodes))

    def _estimate_tokens(self, nodes: List[GmNode]) -> int:
        return sum(len(n.content) + len(n.description) for n in nodes) // 3

    async def sync_embed(self, node: GmNode) -> None:
        """异步同步 embedding，不阻塞主流程"""
        if not self._embed:
            return
        content_hash = hashlib.md5(node.content.encode()).hexdigest()
        if get_vector_hash(self._db, node.id) == content_hash:
            return
        try:
            text = f"{node.name}: {node.description}\n{node.content[:500]}"
            vec = await self._embed(text)
            if vec:
                save_vector(self._db, node.id, node.content, vec)
        except Exception:
            pass
