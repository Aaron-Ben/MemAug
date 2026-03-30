"""
Graph Memory V3 — 图谱维护

执行顺序：
  1. 去重
  2. 全局 PageRank
  3. 社区检测
  4. 社区描述生成
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any, Callable, Coroutine, List, Optional

from ..config import GmConfig
from ..types import MaintenanceResult
from .pagerank import compute_global_page_rank, invalidate_graph_cache
from .community import detect_communities, summarize_communities
from .dedup import dedup

logger = logging.getLogger(__name__)

CompleteFn = Callable[[str, str], Coroutine[Any, Any, str]]
EmbedFn = Callable[[str], Coroutine[Any, Any, List[float]]]


async def run_maintenance(
    db: sqlite3.Connection,
    cfg: GmConfig,
    llm: Optional[CompleteFn] = None,
    embed_fn: Optional[EmbedFn] = None,
) -> MaintenanceResult:
    start = time.time()

    invalidate_graph_cache()

    # 1. 去重
    dedup_result = dedup(db, cfg)

    if dedup_result.merged > 0:
        invalidate_graph_cache()

    # 2. 全局 PageRank
    pagerank_result = compute_global_page_rank(db, cfg)

    # 3. 社区检测
    community_result = detect_communities(db)

    # 4. 社区描述生成
    community_summaries = 0
    if llm and community_result.communities:
        try:
            community_summaries = await summarize_communities(
                db, community_result.communities, llm, embed_fn,
            )
            logger.debug(f"maintenance: generated {community_summaries} community summaries")
        except Exception as err:
            logger.debug(f"maintenance: community summarization failed: {err}")

    return MaintenanceResult(
        dedup=dedup_result,
        pagerank=pagerank_result,
        community=community_result,
        community_summaries=community_summaries,
        duration_ms=int((time.time() - start) * 1000),
    )
