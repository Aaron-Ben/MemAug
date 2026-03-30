"""
Graph Memory V3 — 完整 CRUD 层

节点/边 CRUD、FTS5 搜索、递归 CTE 图遍历、消息/信号 CRUD、向量存储/搜索、社区操作
"""

from __future__ import annotations

import array
import hashlib
import json
import logging
import math
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ..types import (
    CommunitySummary,
    EdgeType,
    GmEdge,
    GmNode,
    NodeType,
    NodeStatus,
    ScoredCommunity,
    ScoredNode,
    Signal,
    SignalType,
)

logger = logging.getLogger(__name__)

# ─── 工具 ─────────────────────────────────────────────────────


def uid(prefix: str) -> str:
    import random
    return f"{prefix}-{int(time.time()*1000)}-{random.randint(10000,99999)}"


def _to_node(r: tuple, columns: tuple) -> GmNode:
    d = dict(zip(columns, r))
    return GmNode(
        id=d["id"],
        type=NodeType(d["type"]),
        name=d["name"],
        description=d["description"] or "",
        content=d["content"],
        status=NodeStatus(d["status"]),
        validated_count=d["validated_count"],
        source_sessions=json.loads(d["source_sessions"] or "[]"),
        community_id=d.get("community_id"),
        pagerank=d.get("pagerank") or 0,
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )


def _to_edge(r: tuple, columns: tuple) -> GmEdge:
    d = dict(zip(columns, r))
    return GmEdge(
        id=d["id"],
        from_id=d["from_id"],
        to_id=d["to_id"],
        type=EdgeType(d["type"]),
        instruction=d["instruction"],
        condition=d.get("condition"),
        session_id=d["session_id"],
        created_at=d["created_at"],
    )


def _node_columns() -> tuple:
    return ("id", "type", "name", "description", "content", "status",
            "validated_count", "source_sessions", "community_id", "pagerank",
            "created_at", "updated_at")


def _edge_columns() -> tuple:
    return ("id", "from_id", "to_id", "type", "instruction", "condition",
            "session_id", "created_at")


def normalize_name(name: str) -> str:
    """标准化 name：全小写，空格转连字符，保留中文"""
    s = name.strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff\-]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    s = s.strip("-")
    return s


def _blob_to_vec(blob: bytes) -> List[float]:
    """BLOB → float list"""
    a = array.array("f")
    a.frombytes(blob)
    return a.tolist()


def _vec_to_blob(vec: List[float]) -> bytes:
    """float list → BLOB"""
    a = array.array("f", vec)
    return a.tobytes()


# ─── 节点 CRUD ───────────────────────────────────────────────


def find_by_name(db: sqlite3.Connection, name: str) -> Optional[GmNode]:
    cols = _node_columns()
    r = db.execute(f"SELECT {','.join(cols)} FROM gm_nodes WHERE name = ?", (normalize_name(name),)).fetchone()
    if not r:
        return None
    return _to_node(r, cols)


def find_by_id(db: sqlite3.Connection, node_id: str) -> Optional[GmNode]:
    cols = _node_columns()
    r = db.execute(f"SELECT {','.join(cols)} FROM gm_nodes WHERE id = ?", (node_id,)).fetchone()
    if not r:
        return None
    return _to_node(r, cols)


def all_active_nodes(db: sqlite3.Connection) -> List[GmNode]:
    cols = _node_columns()
    rows = db.execute(f"SELECT {','.join(cols)} FROM gm_nodes WHERE status='active'").fetchall()
    return [_to_node(r, cols) for r in rows]


def all_edges(db: sqlite3.Connection) -> List[GmEdge]:
    cols = _edge_columns()
    rows = db.execute(f"SELECT {','.join(cols)} FROM gm_edges").fetchall()
    return [_to_edge(r, cols) for r in rows]


def upsert_node(
    db: sqlite3.Connection,
    type_str: str,
    name: str,
    description: str,
    content: str,
    session_id: str,
) -> Tuple[GmNode, bool]:
    """upsert 节点，返回 (node, is_new)"""
    norm_name = normalize_name(name)
    ex = find_by_name(db, norm_name)

    if ex:
        sessions = list(set(ex.source_sessions + [session_id]))
        best_content = content if len(content) > len(ex.content) else ex.content
        best_desc = description if len(description) > len(ex.description) else ex.description
        count = ex.validated_count + 1
        now = int(time.time() * 1000)
        db.execute(
            "UPDATE gm_nodes SET content=?, description=?, validated_count=?, source_sessions=?, updated_at=? WHERE id=?",
            (best_content, best_desc, count, json.dumps(sessions), now, ex.id),
        )
        return GmNode(
            id=ex.id, type=ex.type, name=norm_name, description=best_desc,
            content=best_content, status=ex.status, validated_count=count,
            source_sessions=sessions, community_id=ex.community_id,
            pagerank=ex.pagerank, created_at=ex.created_at, updated_at=now,
        ), False

    node_id = uid("n")
    now = int(time.time() * 1000)
    db.execute(
        "INSERT INTO gm_nodes (id, type, name, description, content, status, validated_count, source_sessions, created_at, updated_at) VALUES (?,?,?,?,?,'active',1,?,?,?)",
        (node_id, type_str, norm_name, description, content, json.dumps([session_id]), now, now),
    )
    return find_by_name(db, norm_name), True  # type: ignore


def deprecate(db: sqlite3.Connection, node_id: str) -> None:
    db.execute("UPDATE gm_nodes SET status='deprecated', updated_at=? WHERE id=?", (int(time.time() * 1000), node_id))


def merge_nodes(db: sqlite3.Connection, keep_id: str, merge_id: str) -> None:
    """合并两个节点：keepId 保留，mergeId 标记 deprecated，边迁移"""
    keep = find_by_id(db, keep_id)
    merge = find_by_id(db, merge_id)
    if not keep or not merge:
        return

    sessions = list(set(keep.source_sessions + merge.source_sessions))
    count = keep.validated_count + merge.validated_count
    content = keep.content if len(keep.content) >= len(merge.content) else merge.content
    desc = keep.description if len(keep.description) >= len(merge.description) else merge.description
    now = int(time.time() * 1000)

    db.execute(
        "UPDATE gm_nodes SET content=?, description=?, validated_count=?, source_sessions=?, updated_at=? WHERE id=?",
        (content, desc, count, json.dumps(sessions), now, keep_id),
    )
    db.execute("UPDATE gm_edges SET from_id=? WHERE from_id=?", (keep_id, merge_id))
    db.execute("UPDATE gm_edges SET to_id=? WHERE to_id=?", (keep_id, merge_id))
    # 删除自环
    db.execute("DELETE FROM gm_edges WHERE from_id = to_id")
    # 删除重复边
    db.execute("""
        DELETE FROM gm_edges WHERE id NOT IN (
            SELECT MIN(id) FROM gm_edges GROUP BY from_id, to_id, type
        )
    """)
    deprecate(db, merge_id)


def update_pageranks(db: sqlite3.Connection, scores: Dict[str, float]) -> None:
    with db:
        for node_id, score in scores.items():
            db.execute("UPDATE gm_nodes SET pagerank=? WHERE id=?", (score, node_id))


def update_communities(db: sqlite3.Connection, labels: Dict[str, str]) -> None:
    with db:
        for node_id, cid in labels.items():
            db.execute("UPDATE gm_nodes SET community_id=? WHERE id=?", (cid, node_id))


# ─── 边 CRUD ─────────────────────────────────────────────────


def upsert_edge(
    db: sqlite3.Connection,
    from_id: str,
    to_id: str,
    type_str: str,
    instruction: str,
    condition: Optional[str],
    session_id: str,
) -> None:
    ex = db.execute("SELECT id FROM gm_edges WHERE from_id=? AND to_id=? AND type=?", (from_id, to_id, type_str)).fetchone()
    if ex:
        db.execute("UPDATE gm_edges SET instruction=? WHERE id=?", (instruction, ex[0]))
        return
    db.execute(
        "INSERT INTO gm_edges (id, from_id, to_id, type, instruction, condition, session_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (uid("e"), from_id, to_id, type_str, instruction, condition, session_id, int(time.time() * 1000)),
    )


# ─── FTS5 搜索 ───────────────────────────────────────────────


_fts5_available: Optional[bool] = None


def _check_fts5(db: sqlite3.Connection) -> bool:
    global _fts5_available
    if _fts5_available is not None:
        return _fts5_available
    try:
        db.execute("SELECT * FROM gm_nodes_fts LIMIT 0")
        _fts5_available = True
    except sqlite3.OperationalError:
        _fts5_available = False
    return _fts5_available


def search_nodes(db: sqlite3.Connection, query: str, limit: int = 6) -> List[GmNode]:
    cols = _node_columns()
    terms = [t for t in query.strip().split() if t][:8]
    if not terms:
        return top_nodes(db, limit)

    if _check_fts5(db):
        try:
            fts_query = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in terms)
            rows = db.execute(
                f"SELECT n.*, rank FROM gm_nodes_fts fts JOIN gm_nodes n ON n.rowid = fts.rowid WHERE gm_nodes_fts MATCH ? AND n.status = 'active' ORDER BY rank LIMIT ?",
                (fts_query, limit),
            ).fetchall()
            if rows:
                return [_to_node(r, cols) for r in rows]
        except sqlite3.OperationalError:
            pass  # FTS 查询失败，降级

    where = " OR ".join(["(name LIKE ? OR description LIKE ? OR content LIKE ?)"] * len(terms))
    likes = [f"%{t}%" for t in terms for _ in range(3)]
    rows = db.execute(
        f"SELECT {','.join(cols)} FROM gm_nodes WHERE status='active' AND ({where}) ORDER BY pagerank DESC, validated_count DESC, updated_at DESC LIMIT ?",
        (*likes, limit),
    ).fetchall()
    return [_to_node(r, cols) for r in rows]


def top_nodes(db: sqlite3.Connection, limit: int = 6) -> List[GmNode]:
    cols = _node_columns()
    rows = db.execute(
        f"SELECT {','.join(cols)} FROM gm_nodes WHERE status='active' ORDER BY pagerank DESC, validated_count DESC, updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_to_node(r, cols) for r in rows]


# ─── 递归 CTE 图遍历 ────────────────────────────────────────


def graph_walk(
    db: sqlite3.Connection,
    seed_ids: List[str],
    max_depth: int,
) -> Tuple[List[GmNode], List[GmEdge]]:
    if not seed_ids:
        return [], []

    placeholders = ",".join(["?"] * len(seed_ids))
    node_cols = _node_columns()
    edge_cols = _edge_columns()

    walk_rows = db.execute(
        f"""
        WITH RECURSIVE walk(node_id, depth) AS (
            SELECT id, 0 FROM gm_nodes WHERE id IN ({placeholders}) AND status='active'
            UNION
            SELECT
                CASE WHEN e.from_id = w.node_id THEN e.to_id ELSE e.from_id END,
                w.depth + 1
            FROM walk w
            JOIN gm_edges e ON (e.from_id = w.node_id OR e.to_id = w.node_id)
            WHERE w.depth < ?
        )
        SELECT DISTINCT node_id FROM walk
        """,
        (*seed_ids, max_depth),
    ).fetchall()

    node_ids = [r[0] for r in walk_rows]
    if not node_ids:
        return [], []

    np = ",".join(["?"] * len(node_ids))
    nodes = [_to_node(r, node_cols) for r in db.execute(
        f"SELECT {','.join(node_cols)} FROM gm_nodes WHERE id IN ({np}) AND status='active'",
        (*node_ids,),
    ).fetchall()]

    edges = [_to_edge(r, edge_cols) for r in db.execute(
        f"SELECT {','.join(edge_cols)} FROM gm_edges WHERE from_id IN ({np}) AND to_id IN ({np})",
        (*node_ids, *node_ids),
    ).fetchall()]

    return nodes, edges


# ─── 按 session 查询 ────────────────────────────────────────


def get_by_session(db: sqlite3.Connection, session_id: str) -> List[GmNode]:
    cols = _node_columns()
    rows = db.execute(
        f"SELECT DISTINCT n.* FROM gm_nodes n, json_each(n.source_sessions) j WHERE j.value = ? AND n.status = 'active'",
        (session_id,),
    ).fetchall()
    return [_to_node(r, cols) for r in rows]


# ─── 消息 CRUD ───────────────────────────────────────────────


def save_message(
    db: sqlite3.Connection,
    sid: str,
    turn: int,
    role: str,
    content: str,
) -> None:
    db.execute(
        "INSERT OR IGNORE INTO gm_messages (id, session_id, turn_index, role, content, created_at) VALUES (?,?,?,?,?,?)",
        (uid("m"), sid, turn, role, content, int(time.time() * 1000)),
    )


def get_unextracted(db: sqlite3.Connection, sid: str, limit: int) -> List[dict]:
    rows = db.execute(
        "SELECT * FROM gm_messages WHERE session_id=? AND extracted=0 ORDER BY turn_index LIMIT ?",
        (sid, limit),
    ).fetchall()
    cols = [d[0] for d in db.execute("SELECT * FROM gm_messages LIMIT 0").description]
    return [dict(zip(cols, r)) for r in rows]


def mark_extracted(db: sqlite3.Connection, sid: str, up_to_turn: int) -> None:
    db.execute("UPDATE gm_messages SET extracted=1 WHERE session_id=? AND turn_index<=?", (sid, up_to_turn))


def get_max_turn(db: sqlite3.Connection, sid: str) -> int:
    row = db.execute("SELECT MAX(turn_index) FROM gm_messages WHERE session_id=?", (sid,)).fetchone()
    return row[0] if row[0] is not None else -1


def get_episodic_messages(
    db: sqlite3.Connection,
    session_ids: List[str],
    near_time: int,
    max_chars: int = 1500,
) -> List[Dict[str, Any]]:
    """溯源选拉：按 session 拉取 user/assistant 核心对话"""
    if not session_ids:
        return []

    results: List[Dict[str, Any]] = []
    used_chars = 0

    for sid in session_ids:
        if used_chars >= max_chars:
            break
        rows = db.execute(
            "SELECT turn_index, role, content, created_at FROM gm_messages WHERE session_id = ? AND role IN ('user', 'assistant') ORDER BY ABS(created_at - ?) ASC LIMIT 6",
            (sid, near_time),
        ).fetchall()

        for r in rows:
            if used_chars >= max_chars:
                break
            text = _extract_text(r[2])
            if not text.strip():
                continue
            truncated = text[:min(len(text), max_chars - used_chars)]
            results.append({
                "sessionId": sid,
                "turnIndex": r[0],
                "role": r[1],
                "text": truncated,
                "createdAt": r[3],
            })
            used_chars += len(truncated)

    return results


def _extract_text(content: str) -> str:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, str):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
            return parsed["content"]
        if isinstance(parsed, list):
            return "\n".join(
                b.get("text", "") for b in parsed if isinstance(b, dict) and b.get("type") == "text"
            )
        return str(parsed)[:300]
    except (json.JSONDecodeError, TypeError):
        return str(content)[:300]


# ─── 信号 CRUD ───────────────────────────────────────────────


def save_signal(db: sqlite3.Connection, sid: str, signal: Signal) -> None:
    db.execute(
        "INSERT INTO gm_signals (id, session_id, turn_index, type, data, created_at) VALUES (?,?,?,?,?,?)",
        (uid("s"), sid, signal.turn_index, signal.type.value, json.dumps(signal.data), int(time.time() * 1000)),
    )


# ─── 统计 ────────────────────────────────────────────────────


def get_stats(db: sqlite3.Connection) -> Dict[str, Any]:
    total_nodes = db.execute("SELECT COUNT(*) FROM gm_nodes WHERE status='active'").fetchone()[0]
    by_type = {}
    for r in db.execute("SELECT type, COUNT(*) FROM gm_nodes WHERE status='active' GROUP BY type"):
        by_type[r[0]] = r[1]
    total_edges = db.execute("SELECT COUNT(*) FROM gm_edges").fetchone()[0]
    by_edge_type = {}
    for r in db.execute("SELECT type, COUNT(*) FROM gm_edges GROUP BY type"):
        by_edge_type[r[0]] = r[1]
    communities = db.execute(
        "SELECT COUNT(DISTINCT community_id) FROM gm_nodes WHERE status='active' AND community_id IS NOT NULL"
    ).fetchone()[0]
    return {"totalNodes": total_nodes, "byType": by_type, "totalEdges": total_edges, "byEdgeType": by_edge_type, "communities": communities}


# ─── 向量存储 + 搜索 ────────────────────────────────────────


def save_vector(db: sqlite3.Connection, node_id: str, content: str, vec: List[float]) -> None:
    content_hash = hashlib.md5(content.encode()).hexdigest()
    blob = _vec_to_blob(vec)
    db.execute(
        "INSERT INTO gm_vectors (node_id, content_hash, embedding) VALUES (?,?,?) ON CONFLICT(node_id) DO UPDATE SET content_hash=excluded.content_hash, embedding=excluded.embedding",
        (node_id, content_hash, blob),
    )


def get_vector_hash(db: sqlite3.Connection, node_id: str) -> Optional[str]:
    row = db.execute("SELECT content_hash FROM gm_vectors WHERE node_id=?", (node_id,)).fetchone()
    return row[0] if row else None


def get_all_vectors(db: sqlite3.Connection) -> List[Dict[str, Any]]:
    """获取所有向量（供去重/聚类用）"""
    rows = db.execute(
        "SELECT v.node_id, v.embedding FROM gm_vectors v JOIN gm_nodes n ON n.id = v.node_id WHERE n.status = 'active'"
    ).fetchall()
    result = []
    for r in rows:
        vec = _blob_to_vec(r[1])
        result.append({"nodeId": r[0], "embedding": vec})
    return result


def vector_search_with_score(db: sqlite3.Connection, query_vec: List[float], limit: int, min_score: float = 0.35) -> List[ScoredNode]:
    cols = _node_columns()
    rows = db.execute(
        "SELECT v.node_id, v.embedding, n.* FROM gm_vectors v JOIN gm_nodes n ON n.id = v.node_id WHERE n.status = 'active'"
    ).fetchall()

    if not rows:
        return []

    q_norm = math.sqrt(sum(x * x for x in query_vec))
    if q_norm == 0:
        return []

    scored = []
    for r in rows:
        node = _to_node(r[2:], cols)
        vec = _blob_to_vec(r[1])
        dot = sum(a * b for a, b in zip(query_vec, vec))
        v_norm = math.sqrt(sum(x * x for x in vec))
        score = dot / (v_norm * q_norm + 1e-9)
        if score > min_score:
            scored.append(ScoredNode(node=node, score=score))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:limit]


def vector_search(db: sqlite3.Connection, query_vec: List[float], limit: int, min_score: float = 0.35) -> List[GmNode]:
    return [s.node for s in vector_search_with_score(db, query_vec, limit, min_score)]


def community_representatives(db: sqlite3.Connection, per_community: int = 2) -> List[GmNode]:
    """每个社区取最近更新的 topN 个节点"""
    cols = _node_columns()
    rows = db.execute(
        f"SELECT {','.join(cols)} FROM gm_nodes WHERE status = 'active' AND community_id IS NOT NULL ORDER BY community_id, updated_at DESC"
    ).fetchall()

    by_community: Dict[str, List[GmNode]] = {}
    for r in rows:
        node = _to_node(r, cols)
        cid = node.community_id  # type: ignore
        if cid not in by_community:
            by_community[cid] = []
        if len(by_community[cid]) < per_community:
            by_community[cid].append(node)

    # 社区按最新更新时间排序
    communities = sorted(
        by_community.items(),
        key=lambda x: max(n.updated_at for n in x[1]),
        reverse=True,
    )

    result = []
    for _, nodes in communities:
        result.extend(nodes)
    return result


# ─── 社区描述 CRUD ──────────────────────────────────────────


def upsert_community_summary(
    db: sqlite3.Connection,
    community_id: str,
    summary: str,
    node_count: int,
    embedding: Optional[List[float]] = None,
) -> None:
    now = int(time.time() * 1000)
    blob = _vec_to_blob(embedding) if embedding else None
    ex = db.execute("SELECT id FROM gm_communities WHERE id=?", (community_id,)).fetchone()
    if ex:
        if blob:
            db.execute("UPDATE gm_communities SET summary=?, node_count=?, embedding=?, updated_at=? WHERE id=?", (summary, node_count, blob, now, community_id))
        else:
            db.execute("UPDATE gm_communities SET summary=?, node_count=?, updated_at=? WHERE id=?", (summary, node_count, now, community_id))
    else:
        db.execute("INSERT INTO gm_communities (id, summary, node_count, embedding, created_at, updated_at) VALUES (?,?,?,?,?,?)", (community_id, summary, node_count, blob, now, now))


def get_community_summary(db: sqlite3.Connection, community_id: str) -> Optional[CommunitySummary]:
    r = db.execute("SELECT id, summary, node_count, created_at, updated_at FROM gm_communities WHERE id=?", (community_id,)).fetchone()
    if not r:
        return None
    return CommunitySummary(id=r[0], summary=r[1], node_count=r[2], created_at=r[3], updated_at=r[4])


def get_all_community_summaries(db: sqlite3.Connection) -> List[CommunitySummary]:
    rows = db.execute("SELECT id, summary, node_count, created_at, updated_at FROM gm_communities ORDER BY node_count DESC").fetchall()
    return [CommunitySummary(id=r[0], summary=r[1], node_count=r[2], created_at=r[3], updated_at=r[4]) for r in rows]


def community_vector_search(db: sqlite3.Connection, query_vec: List[float], min_score: float = 0.15) -> List[ScoredCommunity]:
    rows = db.execute("SELECT id, summary, node_count, embedding FROM gm_communities WHERE embedding IS NOT NULL").fetchall()
    if not rows:
        return []

    q_norm = math.sqrt(sum(x * x for x in query_vec))
    if q_norm == 0:
        return []

    scored = []
    for r in rows:
        vec = _blob_to_vec(r[3])
        dot = sum(a * b for a, b in zip(query_vec, vec))
        v_norm = math.sqrt(sum(x * x for x in vec))
        score = dot / (v_norm * q_norm + 1e-9)
        if score > min_score:
            scored.append(ScoredCommunity(id=r[0], summary=r[1], score=score, node_count=r[2]))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def nodes_by_community_ids(db: sqlite3.Connection, community_ids: List[str], per_community: int = 3) -> List[GmNode]:
    if not community_ids:
        return []
    cols = _node_columns()
    placeholders = ",".join(["?"] * len(community_ids))
    rows = db.execute(
        f"SELECT {','.join(cols)} FROM gm_nodes WHERE community_id IN ({placeholders}) AND status='active' ORDER BY community_id, updated_at DESC",
        (*community_ids,),
    ).fetchall()

    by_community: Dict[str, List[GmNode]] = {}
    for r in rows:
        node = _to_node(r, cols)
        cid = node.community_id  # type: ignore
        if cid not in by_community:
            by_community[cid] = []
        if len(by_community[cid]) < per_community:
            by_community[cid].append(node)

    result = []
    for cid in community_ids:
        members = by_community.get(cid)
        if members:
            result.extend(members)
    return result


def prune_community_summaries(db: sqlite3.Connection) -> int:
    cursor = db.execute("""
        DELETE FROM gm_communities WHERE id NOT IN (
            SELECT DISTINCT community_id FROM gm_nodes WHERE community_id IS NOT NULL AND status='active'
        )
    """)
    return cursor.rowcount
