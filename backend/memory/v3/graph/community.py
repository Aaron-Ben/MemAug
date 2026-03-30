"""
Graph Memory V3 — Label Propagation 社区检测 + LLM 社区摘要
"""

from __future__ import annotations

import logging
import random
import re
import sqlite3
from typing import Any, Callable, Coroutine, Dict, List, Optional

from ..types import CommunityResult
from ..store.store import (
    update_communities,
    upsert_community_summary,
    prune_community_summaries,
)

logger = logging.getLogger(__name__)

CompleteFn = Callable[[str, str], Coroutine[Any, Any, str]]
EmbedFn = Callable[[str], Coroutine[Any, Any, List[float]]]

COMMUNITY_SUMMARY_SYS = """你是知识图谱摘要引擎。根据节点列表，用简短的描述概括这组节点的主题领域。
要求：
- 只返回短语本身，不要解释
- 描述涵盖的工具/技术/任务领域
- 不要使用"社区"这个词"""


def detect_communities(db: sqlite3.Connection, max_iter: int = 50) -> CommunityResult:
    """运行 Label Propagation 并写回 gm_nodes.community_id"""
    node_rows = db.execute("SELECT id FROM gm_nodes WHERE status='active'").fetchall()
    if not node_rows:
        return CommunityResult(labels={}, communities={}, count=0)

    node_ids = [r[0] for r in node_rows]
    node_set = set(node_ids)

    edge_rows = db.execute("SELECT from_id, to_id FROM gm_edges").fetchall()
    adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for e in edge_rows:
        fid, tid = e[0], e[1]
        if fid in node_set and tid in node_set:
            adj[fid].append(tid)
            adj[tid].append(fid)

    # 初始标签：每个节点 = 自己的 ID
    label: Dict[str, str] = {nid: nid for nid in node_ids}

    for _ in range(max_iter):
        changed = False
        shuffled = list(node_ids)
        random.shuffle(shuffled)

        for nid in shuffled:
            neighbors = adj.get(nid, [])
            if not neighbors:
                continue

            freq: Dict[str, int] = {}
            for nb in neighbors:
                lb = label[nb]
                freq[lb] = freq.get(lb, 0) + 1

            best_label = label[nid]
            best_count = 0
            for lb, cnt in freq.items():
                if cnt > best_count or (cnt == best_count and lb < best_label):
                    best_label = lb
                    best_count = cnt

            if label[nid] != best_label:
                label[nid] = best_label
                changed = True

        if not changed:
            break

    # 构建社区映射
    communities: Dict[str, List[str]] = {}
    for nid, cid in label.items():
        communities.setdefault(cid, []).append(nid)

    # 按成员数排序，编号 c-1, c-2, ...
    sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
    rename_map = {old_id: f"c-{i + 1}" for i, (old_id, _) in enumerate(sorted_communities)}

    final_labels = {nid: rename_map.get(lb, lb) for nid, lb in label.items()}
    final_communities: Dict[str, List[str]] = {}
    for old_id, members in communities.items():
        new_id = rename_map.get(old_id, old_id)
        final_communities[new_id] = members

    update_communities(db, final_labels)

    return CommunityResult(labels=final_labels, communities=final_communities, count=len(final_communities))


def get_community_peers(db: sqlite3.Connection, node_id: str, limit: int = 5) -> List[str]:
    """获取同社区的节点 ID 列表"""
    row = db.execute(
        "SELECT community_id FROM gm_nodes WHERE id=? AND status='active'",
        (node_id,),
    ).fetchone()
    if not row or not row[0]:
        return []

    rows = db.execute(
        "SELECT id FROM gm_nodes WHERE community_id=? AND id!=? AND status='active' ORDER BY validated_count DESC, updated_at DESC LIMIT ?",
        (row[0], node_id, limit),
    ).fetchall()
    return [r[0] for r in rows]


async def summarize_communities(
    db: sqlite3.Connection,
    communities: Dict[str, List[str]],
    llm: CompleteFn,
    embed_fn: Optional[EmbedFn] = None,
) -> int:
    """为所有社区生成 LLM 摘要描述 + embedding 向量"""
    prune_community_summaries(db)
    generated = 0

    for community_id, member_ids in communities.items():
        if not member_ids:
            continue

        placeholders = ",".join(["?"] * len(member_ids))
        members = db.execute(
            f"SELECT name, type, description FROM gm_nodes WHERE id IN ({placeholders}) AND status='active' ORDER BY validated_count DESC LIMIT 10",
            (*member_ids,),
        ).fetchall()

        if not members:
            continue

        member_text = "\n".join(f"{m[1]}:{m[0]} — {m[2]}" for m in members)

        try:
            summary_raw = await llm(COMMUNITY_SUMMARY_SYS, f"社区成员：\n{member_text}")

            # 清理摘要
            cleaned = summary_raw.strip()
            cleaned = re.sub(r"<think[\s\S]*?(?:<\/think>|$)", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^["\'「」]|["\'「」]$', "", cleaned)
            cleaned = cleaned.replace("\n", " ")
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()[:100]

            if not cleaned:
                continue

            embedding = None
            if embed_fn:
                try:
                    embed_text = f"{cleaned}\n{', '.join(m[0] for m in members)}"
                    embedding = await embed_fn(embed_text)
                except Exception:
                    logger.debug(f"community embedding failed for {community_id}")

            upsert_community_summary(db, community_id, cleaned, len(member_ids), embedding)
            generated += 1
        except Exception as err:
            logger.warning(f"community summary failed for {community_id}: {err}")

    return generated
