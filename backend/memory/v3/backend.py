"""
Graph Memory V3 — MemoryBackend 实现

知识图谱记忆系统：LLM 提取知识三元组 → SQLite 图存储 → 双路径召回 → PPR 排序
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from memory.factory import MemoryBackend

from .config import GmConfig
from .store.db import get_db, close_db
from .store.store import (
    find_by_name,
    upsert_node,
    upsert_edge,
    save_message,
    get_unextracted,
    mark_extracted,
    get_max_turn,
    get_by_session,
    all_active_nodes,
    all_edges,
    get_stats,
    top_nodes,
    normalize_name,
)
from .extractor.extract import Extractor, CompleteFn
from .recaller.recall import Recaller
from .graph.maintenance import run_maintenance
from .format.assemble import assemble_context

logger = logging.getLogger(__name__)


class MemoryV3Backend(MemoryBackend):
    """Graph Memory V3 Backend"""

    def __init__(self):
        self._cfg = GmConfig.from_env()
        self._dbs: Dict[str, sqlite3.Connection] = {}  # character_id -> db
        self._extractors: Dict[str, Extractor] = {}
        self._recallers: Dict[str, Recaller] = {}
        self._turn_counters: Dict[str, int] = {}  # session_id -> count
        self._extract_chains: Dict[str, Any] = {}  # session_id -> asyncio.Lock
        self._embed_service = None
        self._llm_service = None

    @property
    def name(self) -> str:
        return "v3"

    async def initialize(self, app) -> None:
        """初始化，复用 app 上的 EmbeddingService 和 LLM"""
        from app.services.embedding import EmbeddingService
        from app.services.llm import LLM

        try:
            self._embed_service = EmbeddingService()
        except Exception as e:
            logger.warning(f"[v3] EmbeddingService init failed: {e}")

        try:
            self._llm_service = LLM()
        except Exception as e:
            logger.warning(f"[v3] LLM init failed: {e}")

        logger.info("[v3] Graph Memory Backend initialized")

    def _get_db(self, character_id: str) -> sqlite3.Connection:
        if character_id not in self._dbs:
            db_path = self._cfg.db_path.format(character_id=character_id)
            self._dbs[character_id] = get_db(db_path)
        return self._dbs[character_id]

    def _get_extractor(self, character_id: str) -> Extractor:
        if character_id not in self._extractors:
            async def llm_fn(system: str, user: str) -> str:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                return await self._llm_service.generate_response_async(messages)

            self._extractors[character_id] = Extractor(llm_fn)
        return self._extractors[character_id]

    def _get_recaller(self, character_id: str) -> Recaller:
        if character_id not in self._recallers:
            db = self._get_db(character_id)
            recaller = Recaller(db, self._cfg)

            async def embed_fn(text: str) -> List[float]:
                vec = await self._embed_service.get_single_embedding(text)
                return vec or []

            if self._embed_service:
                recaller.set_embed_fn(embed_fn)
            self._recallers[character_id] = recaller
        return self._recallers[character_id]

    # ─── MemoryBackend 接口实现 ─────────────────────────────

    async def search(self, query: str, character_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索记忆：使用 recaller 的双路径召回"""
        if not character_id:
            return []

        recaller = self._get_recaller(character_id)
        result = await recaller.recall(query)

        # 组装为上下文 XML
        db = self._get_db(character_id)
        active_nodes = []
        active_edges = []
        context = assemble_context(db, active_nodes, active_edges, result.nodes, result.edges)

        return [{
            "xml": context["xml"],
            "system_prompt": context["system_prompt"],
            "tokens": context["tokens"],
            "episodic_xml": context["episodic_xml"],
            "nodes": [{"id": n.id, "type": n.type.value, "name": n.name, "content": n.content} for n in result.nodes],
            "edges": [{"from": e.from_id, "to": e.to_id, "type": e.type.value, "instruction": e.instruction} for e in result.edges],
        }]

    async def save_memory(self, character_id: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """保存记忆：提取知识三元组"""
        db = self._get_db(character_id)
        extractor = self._get_extractor(character_id)
        meta = metadata or {}
        session_id = meta.get("session_id", "default")

        # 获取已有节点名列表
        active = await asyncio.to_thread(all_active_nodes, db)
        existing_names = [n.name for n in active]

        # 构造消息
        messages = [{"role": meta.get("role", "user"), "content": content, "turn_index": meta.get("turn_index", 0)}]

        result = await extractor.extract(messages, existing_names)

        created_nodes = []
        created_edges = []

        for nc in result.nodes:
            node, is_new = await asyncio.to_thread(
                upsert_node, db, nc.type.value, nc.name, nc.description, nc.content, session_id
            )
            created_nodes.append({"name": node.name, "type": node.type.value, "is_new": is_new})

            # 异步同步 embedding
            if self._embed_service:
                recaller = self._get_recaller(character_id)
                asyncio.create_task(recaller.sync_embed(node))

        for ec in result.edges:
            from_node = await asyncio.to_thread(find_by_name, db, ec.from_name)
            to_node = await asyncio.to_thread(find_by_name, db, ec.to_name)
            if from_node and to_node:
                await asyncio.to_thread(
                    upsert_edge, db, from_node.id, to_node.id, ec.type.value, ec.instruction, ec.condition, session_id
                )
                created_edges.append({"from": ec.from_name, "to": ec.to_name, "type": ec.type.value})

        return {
            "nodes": created_nodes,
            "edges": created_edges,
            "session_id": session_id,
        }

    async def get_recent_memories(self, character_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的知识节点"""
        db = self._get_db(character_id)
        nodes = await asyncio.to_thread(top_nodes, db, limit)
        return [
            {
                "id": n.id,
                "type": n.type.value,
                "name": n.name,
                "description": n.description,
                "content": n.content[:500],
                "validated_count": n.validated_count,
                "pagerank": n.pagerank,
            }
            for n in nodes
        ]

    # ─── 图谱专用方法 ───────────────────────────────────────

    async def ingest_message(
        self,
        character_id: str,
        session_id: str,
        role: str,
        content: str,
        turn_index: Optional[int] = None,
    ) -> None:
        """保存消息并触发异步提取"""
        db = self._get_db(character_id)

        if turn_index is None:
            max_turn = await asyncio.to_thread(get_max_turn, db, session_id)
            turn_index = max_turn + 1

        await asyncio.to_thread(save_message, db, session_id, turn_index, role, content)

        # 更新 turn counter
        key = session_id
        self._turn_counters[key] = self._turn_counters.get(key, 0) + 1

        # 每 compactTurnCount 轮触发一次提取
        if self._turn_counters[key] >= self._cfg.compact_turn_count:
            self._turn_counters[key] = 0
            asyncio.create_task(self._run_turn_extract(character_id, session_id))

    async def _run_turn_extract(self, character_id: str, session_id: str) -> None:
        """异步提取：从未提取的消息中抽取知识三元组"""
        db = self._get_db(character_id)
        extractor = self._get_extractor(character_id)

        # 确保同一个 session 串行执行
        if session_id not in self._extract_chains:
            self._extract_chains[session_id] = asyncio.Lock()

        async with self._extract_chains[session_id]:
            try:
                messages = await asyncio.to_thread(get_unextracted, db, session_id, 20)
                if not messages:
                    return

                max_turn = max(m["turn_index"] for m in messages)

                active = await asyncio.to_thread(all_active_nodes, db)
                existing_names = [n.name for n in active]

                result = await extractor.extract(messages, existing_names)

                for nc in result.nodes:
                    node, _ = await asyncio.to_thread(
                        upsert_node, db, nc.type.value, nc.name, nc.description, nc.content, session_id
                    )
                    if self._embed_service:
                        recaller = self._get_recaller(character_id)
                        await recaller.sync_embed(node)

                for ec in result.edges:
                    from_node = await asyncio.to_thread(find_by_name, db, ec.from_name)
                    to_node = await asyncio.to_thread(find_by_name, db, ec.to_name)
                    if from_node and to_node:
                        await asyncio.to_thread(
                            upsert_edge, db, from_node.id, to_node.id, ec.type.value, ec.instruction, ec.condition, session_id
                        )

                await asyncio.to_thread(mark_extracted, db, session_id, max_turn)

            except Exception as e:
                logger.error(f"[v3] turn extract failed for session={session_id}: {e}")

    async def finalize_session(self, character_id: str, session_id: str) -> Dict[str, Any]:
        """Session 结束：finalize + 维护"""
        db = self._get_db(character_id)
        extractor = self._get_extractor(character_id)

        # finalize：EVENT→PATTERN 升级、补充关系
        session_nodes = await asyncio.to_thread(get_by_session, db, session_id)
        if session_nodes:
            active = await asyncio.to_thread(all_active_nodes, db)
            summary = f"Total {len(active)} nodes, {len(await asyncio.to_thread(all_edges, db))} edges"
            finalize_result = await extractor.finalize(
                [{"id": n.id, "type": n.type.value, "name": n.name, "description": n.description, "validated_count": n.validated_count} for n in session_nodes],
                summary,
            )

            # 处理 promoted patterns (EVENT→PATTERN 升级)
            for ps in finalize_result.promoted_patterns:
                await asyncio.to_thread(
                    upsert_node, db, "PATTERN", ps.name, ps.description, ps.content, session_id
                )

            # 处理 new edges
            for ec in finalize_result.new_edges:
                from_node = await asyncio.to_thread(find_by_name, db, ec.from_name)
                to_node = await asyncio.to_thread(find_by_name, db, ec.to_name)
                if from_node and to_node:
                    await asyncio.to_thread(
                        upsert_edge, db, from_node.id, to_node.id, ec.type.value, ec.instruction, ec.condition, session_id
                    )

            # 处理 invalidations
            from .store.store import deprecate
            for node_id in finalize_result.invalidations:
                await asyncio.to_thread(deprecate, db, node_id)

        # 维护
        async def llm_fn(system: str, user: str) -> str:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return await self._llm_service.generate_response_async(messages)

        async def embed_fn(text: str) -> List[float]:
            vec = await self._embed_service.get_single_embedding(text)
            return vec or []

        maintenance_result = await run_maintenance(
            db, self._cfg,
            llm=llm_fn if self._llm_service else None,
            embed_fn=embed_fn if self._embed_service else None,
        )

        return {
            "finalized_nodes": len(session_nodes),
            "maintenance": {
                "dedup_merged": maintenance_result.dedup.merged,
                "pagerank_top": maintenance_result.pagerank.top_k[:5],
                "communities": maintenance_result.community.count,
                "community_summaries": maintenance_result.community_summaries,
                "duration_ms": maintenance_result.duration_ms,
            },
        }

    async def get_graph_stats(self, character_id: str) -> Dict[str, Any]:
        """获取图谱统计"""
        db = self._get_db(character_id)
        return await asyncio.to_thread(get_stats, db)
