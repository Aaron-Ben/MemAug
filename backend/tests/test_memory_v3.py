"""
Graph Memory V3 — 测试套件

覆盖：Store CRUD、FTS5 搜索、图遍历、PageRank、社区检测、去重、提取器解析、上下文组装、召回
所有测试使用内存 SQLite，无需外部服务。
"""

from __future__ import annotations

import array
import json
import time
import sqlite3
from typing import List, Optional

import pytest

# ─── sys.path 设置 ──────────────────────────────────────────

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory.v3.store.db import get_db, close_db
from memory.v3.store import store
from memory.v3.config import GmConfig
from memory.v3.types import (
    EdgeType, GmEdge, GmNode, NodeType, NodeStatus,
    ExtractionResult, NodeCandidate, EdgeCandidate,
)
from memory.v3.extractor.extract import Extractor, _extract_json, _correct_edge_type, normalize_name
from memory.v3.graph.pagerank import (
    personalized_page_rank,
    compute_global_page_rank,
    invalidate_graph_cache,
)
from memory.v3.graph.community import detect_communities, get_community_peers
from memory.v3.graph.dedup import dedup, detect_duplicates, _cosine_sim
from memory.v3.graph.maintenance import run_maintenance
from memory.v3.recaller.recall import Recaller
from memory.v3.format.assemble import (
    assemble_context,
    build_system_prompt_addition,
)
from memory.v3.backend import MemoryV3Backend


# ─── Fixtures ───────────────────────────────────────────────


@pytest.fixture
def db():
    """每个测试独立的内存数据库"""
    conn = get_db(":memory:")
    yield conn
    close_db(conn)


@pytest.fixture
def cfg():
    return GmConfig()


def insert_node(
    db: sqlite3.Connection,
    *,
    name: str,
    type_: str = "TOPIC",
    description: str = "",
    content: str = "",
    status: str = "active",
    validated_count: int = 1,
    sessions: Optional[List[str]] = None,
    community_id: Optional[str] = None,
) -> str:
    """快速插入测试节点"""
    from memory.v3.store.store import uid
    node_id = uid("n")
    now = int(time.time() * 1000)
    db.execute(
        "INSERT INTO gm_nodes (id, type, name, description, content, status, validated_count, source_sessions, community_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (node_id, type_, name, description or f"desc of {name}", content or f"content of {name}", status, validated_count, json.dumps(sessions or ["test-session"]), community_id, now, now),
    )
    return node_id


def insert_edge(
    db: sqlite3.Connection,
    *,
    from_id: str,
    to_id: str,
    type_: str = "RELATED_TO",
    instruction: str = "test instruction",
) -> None:
    """快速插入测试边"""
    from memory.v3.store.store import uid
    db.execute(
        "INSERT INTO gm_edges (id, from_id, to_id, type, instruction, session_id, created_at) VALUES (?,?,?,?,?,?,?)",
        (uid("e"), from_id, to_id, type_, instruction, "test-session", int(time.time() * 1000)),
    )


def insert_vector(db: sqlite3.Connection, node_id: str, vec: List[float]) -> None:
    """插入测试向量"""
    blob = array.array("f", vec).tobytes()
    db.execute(
        "INSERT INTO gm_vectors (node_id, content_hash, embedding) VALUES (?,?,?)",
        (node_id, "test-hash", blob),
    )


# ═════════════════════════════════════════════════════════════
#  1. Store — 节点 CRUD
# ═════════════════════════════════════════════════════════════


class TestNodeCRUD:
    def test_upsert_new_node(self, db):
        node, is_new = store.upsert_node(db, "PERSON", "mom", "user's mother", "content here", "sess-1")
        assert is_new is True
        assert node.name == "mom"
        assert node.type == NodeType.PERSON
        assert node.validated_count == 1
        assert "sess-1" in node.source_sessions

    def test_upsert_merge_existing(self, db):
        n1, _ = store.upsert_node(db, "TOPIC", "anime", "short desc", "short content", "sess-1")
        n2, is_new = store.upsert_node(db, "TOPIC", "anime", "longer description here", "much longer content here", "sess-2")
        assert is_new is False
        assert n2.validated_count == 2
        assert "sess-1" in n2.source_sessions
        assert "sess-2" in n2.source_sessions
        assert len(n2.content) > len("short content")

    def test_find_by_name(self, db):
        store.upsert_node(db, "EVENT", "graduation-trip", "graduation trip", "content", "s1")
        found = store.find_by_name(db, "graduation-trip")
        assert found is not None
        assert found.type == NodeType.EVENT

    def test_find_by_name_not_found(self, db):
        assert store.find_by_name(db, "nonexistent") is None

    def test_find_by_id(self, db):
        node, _ = store.upsert_node(db, "PATTERN", "exam-anxiety", "desc", "content", "s1")
        found = store.find_by_id(db, node.id)
        assert found is not None
        assert found.name == "exam-anxiety"

    def test_deprecate(self, db):
        node, _ = store.upsert_node(db, "PREFERENCE", "like-encouragement", "desc", "content", "s1")
        store.deprecate(db, node.id)
        found = store.find_by_id(db, node.id)
        assert found.status == NodeStatus.DEPRECATED

    def test_merge_nodes(self, db):
        keep_id = insert_node(db, name="topic-a", type_="TOPIC", validated_count=5, content="keep content")
        merge_id = insert_node(db, name="topic-b", type_="TOPIC", validated_count=2, content="merge content longer")

        store.merge_nodes(db, keep_id, merge_id)

        keep = store.find_by_id(db, keep_id)
        merged = store.find_by_id(db, merge_id)

        assert keep.validated_count == 5 + 2
        assert merged.status == NodeStatus.DEPRECATED

    def test_all_active_nodes(self, db):
        insert_node(db, name="active-1")
        insert_node(db, name="active-2")
        insert_node(db, name="deprecated-1", status="deprecated")

        active = store.all_active_nodes(db)
        assert len(active) == 2

    def test_upsert_user_node(self, db):
        node, is_new = store.upsert_node(db, "USER", "user-profile", "user profile", "基本信息: 25岁\n性格特点: 内向", "sess-1")
        assert is_new is True
        assert node.type == NodeType.USER

    def test_upsert_case_node(self, db):
        node, is_new = store.upsert_node(db, "CASE", "insomnia-case", "insomnia problem", "问题: 失眠\n原因: 压力大\n方案: 运动", "sess-1")
        assert is_new is True
        assert node.type == NodeType.CASE

    def test_normalize_name(self):
        assert normalize_name("My Mom") == "my-mom"
        assert normalize_name("  Hello World  ") == "hello-world"
        assert normalize_name("exam_anxiety") == "exam-anxiety"
        assert normalize_name("中文测试") == "中文测试"


# ═════════════════════════════════════════════════════════════
#  2. Store — 边 CRUD
# ═════════════════════════════════════════════════════════════


class TestEdgeCRUD:
    def test_upsert_new_edge(self, db):
        n1 = insert_node(db, name="person-1", type_="PERSON")
        n2 = insert_node(db, name="event-1", type_="EVENT")

        store.upsert_edge(db, n1, n2, "INVOLVED_IN", "参与了该事件", None, "sess-1")

        edges = store.all_edges(db)
        assert len(edges) == 1
        assert edges[0].type == EdgeType.INVOLVED_IN

    def test_upsert_merge_edge(self, db):
        n1 = insert_node(db, name="person-1", type_="PERSON")
        n2 = insert_node(db, name="event-1", type_="EVENT")

        store.upsert_edge(db, n1, n2, "INVOLVED_IN", "old instruction", None, "s1")
        store.upsert_edge(db, n1, n2, "INVOLVED_IN", "new instruction", None, "s1")

        edges = store.all_edges(db)
        assert len(edges) == 1
        assert edges[0].instruction == "new instruction"

    def test_upsert_has_preference_edge(self, db):
        u1 = insert_node(db, name="user-profile", type_="USER")
        p1 = insert_node(db, name="prefer-encouragement", type_="PREFERENCE")

        store.upsert_edge(db, u1, p1, "HAS_PREFERENCE", "偏好鼓励式回应", None, "sess-1")

        edges = store.all_edges(db)
        assert len(edges) == 1
        assert edges[0].type == EdgeType.HAS_PREFERENCE

    def test_upsert_resolved_by_edge(self, db):
        c1 = insert_node(db, name="insomnia-case", type_="CASE")
        e1 = insert_node(db, name="therapy-session", type_="EVENT")

        store.upsert_edge(db, c1, e1, "RESOLVED_BY", "通过心理咨询解决", None, "sess-1")

        edges = store.all_edges(db)
        assert len(edges) == 1
        assert edges[0].type == EdgeType.RESOLVED_BY


# ═════════════════════════════════════════════════════════════
#  3. Store — FTS5 搜索
# ═════════════════════════════════════════════════════════════


class TestFTS5Search:
    def test_search_by_keyword(self, db):
        insert_node(db, name="mom", content="妈妈最近身体不太好，需要多关心")
        insert_node(db, name="anime", content="喜欢看进击的巨人")

        results = store.search_nodes(db, "妈妈", 5)
        assert len(results) >= 1
        assert any("mom" in r.name for r in results)

    def test_search_empty_query_returns_top(self, db):
        insert_node(db, name="a", validated_count=10)
        insert_node(db, name="b", validated_count=3)

        results = store.search_nodes(db, "", 5)
        assert len(results) == 2
        assert results[0].validated_count >= results[1].validated_count

    def test_search_no_match(self, db):
        insert_node(db, name="test-node", content="some content")
        results = store.search_nodes(db, "zzzzzznonexistent", 5)
        assert len(results) == 0


# ═════════════════════════════════════════════════════════════
#  4. Store — 图遍历
# ═════════════════════════════════════════════════════════════


class TestGraphWalk:
    def test_walk_simple_chain(self, db):
        n1 = insert_node(db, name="person-1", type_="PERSON")
        n2 = insert_node(db, name="event-1", type_="EVENT")
        n3 = insert_node(db, name="pattern-1", type_="PATTERN")

        insert_edge(db, from_id=n1, to_id=n2, type_="INVOLVED_IN")
        insert_edge(db, from_id=n2, to_id=n3, type_="TRIGGERS")

        nodes, edges = store.graph_walk(db, [n1], 2)
        assert len(nodes) == 3
        assert len(edges) == 2

    def test_walk_empty_seeds(self, db):
        nodes, edges = store.graph_walk(db, [], 2)
        assert nodes == []
        assert edges == []

    def test_walk_depth_limit(self, db):
        n1 = insert_node(db, name="a")
        n2 = insert_node(db, name="b")
        n3 = insert_node(db, name="c")
        n4 = insert_node(db, name="d")

        insert_edge(db, from_id=n1, to_id=n2)
        insert_edge(db, from_id=n2, to_id=n3)
        insert_edge(db, from_id=n3, to_id=n4)

        # depth=1: only n1 and its direct neighbors
        nodes, _ = store.graph_walk(db, [n1], 1)
        node_names = {n.name for n in nodes}
        assert "a" in node_names
        assert "b" in node_names
        assert "c" not in node_names


# ═════════════════════════════════════════════════════════════
#  5. Store — 消息
# ═════════════════════════════════════════════════════════════


class TestMessages:
    def test_save_and_get_unextracted(self, db):
        store.save_message(db, "sess-1", 0, "user", "hello")
        store.save_message(db, "sess-1", 1, "assistant", "hi there")

        msgs = store.get_unextracted(db, "sess-1", 10)
        assert len(msgs) == 2

    def test_mark_extracted(self, db):
        store.save_message(db, "sess-1", 0, "user", "hello")
        store.save_message(db, "sess-1", 1, "assistant", "hi")
        store.save_message(db, "sess-1", 2, "user", "next")

        store.mark_extracted(db, "sess-1", 1)

        remaining = store.get_unextracted(db, "sess-1", 10)
        assert len(remaining) == 1
        assert remaining[0]["turn_index"] == 2

    def test_get_max_turn(self, db):
        assert store.get_max_turn(db, "sess-1") == -1
        store.save_message(db, "sess-1", 0, "user", "a")
        store.save_message(db, "sess-1", 5, "assistant", "b")
        assert store.get_max_turn(db, "sess-1") == 5

    def test_get_by_session(self, db):
        insert_node(db, name="topic-a", sessions=["sess-1", "sess-2"])
        insert_node(db, name="topic-b", sessions=["sess-2"])

        s1_nodes = store.get_by_session(db, "sess-1")
        s2_nodes = store.get_by_session(db, "sess-2")

        assert len(s1_nodes) == 1
        assert len(s2_nodes) == 2

    def test_get_stats(self, db):
        insert_node(db, name="person-1", type_="PERSON")
        insert_node(db, name="topic-1", type_="TOPIC")
        insert_node(db, name="event-1", type_="EVENT")
        n4 = insert_node(db, name="deprecated-1", type_="TOPIC", status="deprecated")

        stats = store.get_stats(db)
        assert stats["totalNodes"] == 3  # deprecated excluded
        assert stats["byType"]["PERSON"] == 1
        assert stats["byType"]["TOPIC"] == 1
        assert stats["byType"]["EVENT"] == 1


# ═════════════════════════════════════════════════════════════
#  6. Store — 向量
# ═════════════════════════════════════════════════════════════


class TestVectors:
    def test_save_and_search_vector(self, db):
        n1 = insert_node(db, name="mom", content="妈妈最近身体不太好")
        n2 = insert_node(db, name="anime", content="喜欢看进击的巨人")

        # n1 points right, n2 points up — query right should match n1
        insert_vector(db, n1, [1.0, 0.0, 0.0])
        insert_vector(db, n2, [0.0, 1.0, 0.0])

        results = store.vector_search_with_score(db, [1.0, 0.0, 0.0], 5, min_score=0.5)
        assert len(results) >= 1
        assert results[0].node.name == "mom"
        assert results[0].score > 0.9

    def test_vector_hash(self, db):
        n1 = insert_node(db, name="test-node")
        store.save_vector(db, n1, "test content", [1.0, 2.0, 3.0])

        h = store.get_vector_hash(db, n1)
        assert h is not None
        assert len(h) == 32  # md5 hex

    def test_get_all_vectors(self, db):
        n1 = insert_node(db, name="a")
        n2 = insert_node(db, name="b")
        insert_vector(db, n1, [1.0, 0.0])
        insert_vector(db, n2, [0.0, 1.0])

        all_vecs = store.get_all_vectors(db)
        assert len(all_vecs) == 2


# ═════════════════════════════════════════════════════════════
#  7. PageRank
# ═════════════════════════════════════════════════════════════


class TestPageRank:
    def test_personalized_page_rank(self, db, cfg):
        n1 = insert_node(db, name="seed-1")
        n2 = insert_node(db, name="neighbor")
        n3 = insert_node(db, name="far-away")

        insert_edge(db, from_id=n1, to_id=n2)
        insert_edge(db, from_id=n2, to_id=n3)

        invalidate_graph_cache()
        scores = personalized_page_rank(db, [n1], [n1, n2, n3], cfg)

        assert scores[n1] > scores[n3]  # seed should score higher than far node
        assert scores[n1] > 0
        assert scores[n2] > 0

    def test_ppr_empty_seeds(self, db, cfg):
        scores = personalized_page_rank(db, [], ["any"], cfg)
        assert scores == {}

    def test_global_page_rank(self, db, cfg):
        n1 = insert_node(db, name="hub", validated_count=10)
        n2 = insert_node(db, name="spoke-1")
        n3 = insert_node(db, name="spoke-2")

        insert_edge(db, from_id=n1, to_id=n2)
        insert_edge(db, from_id=n1, to_id=n3)

        invalidate_graph_cache()
        result = compute_global_page_rank(db, cfg)

        assert len(result.scores) == 3
        assert len(result.top_k) <= 20
        # hub connected to both spokes should rank high
        assert result.top_k[0]["name"] == "hub"

        # Check DB was updated
        node = store.find_by_id(db, n1)
        assert node.pagerank > 0


# ═════════════════════════════════════════════════════════════
#  8. Community Detection
# ═════════════════════════════════════════════════════════════


class TestCommunity:
    def test_detect_communities_connected(self, db):
        n1 = insert_node(db, name="a")
        n2 = insert_node(db, name="b")
        n3 = insert_node(db, name="c")
        n4 = insert_node(db, name="isolate")

        insert_edge(db, from_id=n1, to_id=n2)
        insert_edge(db, from_id=n2, to_id=n3)

        result = detect_communities(db)
        assert result.count >= 1

        # a, b, c should be in same community
        cid_a = result.labels.get(n1)
        cid_b = result.labels.get(n2)
        cid_c = result.labels.get(n3)
        assert cid_a == cid_b == cid_c

    def test_detect_communities_empty(self, db):
        result = detect_communities(db)
        assert result.count == 0

    def test_community_peers(self, db):
        n1 = insert_node(db, name="a", community_id="c-1")
        n2 = insert_node(db, name="b", community_id="c-1")
        n3 = insert_node(db, name="c", community_id="c-1")
        n4 = insert_node(db, name="d", community_id="c-2")

        peers = get_community_peers(db, n1, 5)
        assert n2 in peers
        assert n3 in peers
        assert n4 not in peers

    def test_community_peers_no_community(self, db):
        n1 = insert_node(db, name="loner")
        peers = get_community_peers(db, n1, 5)
        assert peers == []


# ═════════════════════════════════════════════════════════════
#  9. Dedup
# ═════════════════════════════════════════════════════════════


class TestDedup:
    def test_cosine_similarity(self):
        assert abs(_cosine_sim([1, 0, 0], [1, 0, 0]) - 1.0) < 0.001
        assert abs(_cosine_sim([1, 0, 0], [0, 1, 0])) < 0.001
        assert abs(_cosine_sim([1, 1, 0], [1, 0, 0]) - 0.707) < 0.01

    def test_detect_duplicates(self, db, cfg):
        n1 = insert_node(db, name="mom-health", type_="EVENT")
        n2 = insert_node(db, name="mother-illness", type_="EVENT")

        # Same direction vectors = high similarity
        insert_vector(db, n1, [1.0, 0.0, 0.0])
        insert_vector(db, n2, [0.95, 0.05, 0.0])

        pairs = detect_duplicates(db, cfg)
        assert len(pairs) >= 1
        assert pairs[0]["similarity"] > 0.9

    def test_dedup_merges_same_type(self, db, cfg):
        n1 = insert_node(db, name="topic-a", type_="TOPIC", validated_count=5)
        n2 = insert_node(db, name="topic-b", type_="TOPIC", validated_count=2)

        insert_vector(db, n1, [1.0, 0.0])
        insert_vector(db, n2, [0.99, 0.01])

        invalidate_graph_cache()
        result = dedup(db, cfg)
        assert result.merged == 1

        keep = store.find_by_id(db, n1)
        merged = store.find_by_id(db, n2)
        assert keep.status == NodeStatus.ACTIVE  # higher validated_count
        assert merged.status == NodeStatus.DEPRECATED

    def test_dedup_no_cross_type(self, db, cfg):
        n1 = insert_node(db, name="a", type_="PERSON")
        n2 = insert_node(db, name="b", type_="TOPIC")

        insert_vector(db, n1, [1.0, 0.0])
        insert_vector(db, n2, [0.99, 0.01])

        result = dedup(db, cfg)
        assert result.merged == 0  # different types, no merge


# ═════════════════════════════════════════════════════════════
#  10. Extractor — JSON 解析与边修正
# ═════════════════════════════════════════════════════════════


class TestExtractor:
    def test_extract_json_clean(self):
        raw = '{"nodes":[],"edges":[]}'
        assert _extract_json(raw) == raw

    def test_extract_json_with_markdown(self):
        raw = '```json\n{"nodes":[]}\n```'
        assert _extract_json(raw) == '{"nodes":[]}'

    def test_extract_json_embedded(self):
        raw = 'Here is the result: {"nodes":[{"type":"PERSON","name":"test","content":"c"}]} done'
        result = _extract_json(raw)
        assert json.loads(result)["nodes"][0]["type"] == "PERSON"

    def test_edge_correction_person_event(self):
        edge = {"from": "my-mom", "to": "graduation", "type": "RELATED_TO", "instruction": "test"}
        name_map = {"my-mom": "PERSON", "graduation": "EVENT"}
        corrected = _correct_edge_type(edge, name_map)
        assert corrected is not None
        assert corrected["type"] == "INVOLVED_IN"  # PERSON→EVENT forced to INVOLVED_IN

    def test_edge_correction_topic_pattern(self):
        edge = {"from": "exam-topic", "to": "exam-anxiety", "type": "RELATED_TO", "instruction": "test"}
        name_map = {"exam-topic": "TOPIC", "exam-anxiety": "PATTERN"}
        corrected = _correct_edge_type(edge, name_map)
        assert corrected is not None
        assert corrected["type"] == "TRIGGERS"  # TOPIC→PATTERN forced to TRIGGERS

    def test_edge_correction_event_event(self):
        edge = {"from": "breakup", "to": "move-city", "type": "RELATED_TO", "instruction": "test"}
        name_map = {"breakup": "EVENT", "move-city": "EVENT"}
        corrected = _correct_edge_type(edge, name_map)
        assert corrected is not None
        assert corrected["type"] == "LEADS_TO"  # EVENT→EVENT forced to LEADS_TO

    def test_edge_correction_bad_direction(self, db):
        edge = {"from": "a", "to": "b", "type": "INVOLVED_IN", "instruction": "test"}
        name_map = {"a": "TOPIC", "b": "PERSON"}  # TOPIC→PERSON with INVOLVED_IN is invalid
        assert _correct_edge_type(edge, name_map) is None

    def test_edge_correction_user_preference(self):
        edge = {"from": "user-profile", "to": "prefer-encouragement", "type": "CARES_ABOUT", "instruction": "test"}
        name_map = {"user-profile": "USER", "prefer-encouragement": "PREFERENCE"}
        corrected = _correct_edge_type(edge, name_map)
        assert corrected is not None
        assert corrected["type"] == "HAS_PREFERENCE"

    def test_edge_correction_case_event(self):
        edge = {"from": "insomnia-case", "to": "therapy-session", "type": "RELATED_TO", "instruction": "test"}
        name_map = {"insomnia-case": "CASE", "therapy-session": "EVENT"}
        corrected = _correct_edge_type(edge, name_map)
        assert corrected is not None
        assert corrected["type"] == "RESOLVED_BY"

    def test_edge_correction_case_pattern(self):
        edge = {"from": "burnout-case", "to": "exercise-routine", "type": "RELATED_TO", "instruction": "test"}
        name_map = {"burnout-case": "CASE", "exercise-routine": "PATTERN"}
        corrected = _correct_edge_type(edge, name_map)
        assert corrected is not None
        assert corrected["type"] == "RESOLVED_BY"

    @pytest.mark.asyncio
    async def test_extractor_parse_full_response(self):
        """测试 Extractor 解析完整的 LLM 响应"""
        async def mock_llm(system: str, user: str) -> str:
            return json.dumps({
                "nodes": [
                    {"type": "PERSON", "name": "mom", "description": "user's mother", "content": "mom\n关系: 母亲\n描述: 很关心用户"},
                    {"type": "EVENT", "name": "mom-surgery", "description": "妈妈做手术", "content": "mom-surgery\n时间: 上周\n经过: 住院手术"},
                ],
                "edges": [
                    {"from": "mom", "to": "mom-surgery", "type": "INVOLVED_IN", "instruction": "妈妈经历了手术"},
                ],
            })

        extractor = Extractor(mock_llm)
        result = await extractor.extract(
            [{"role": "user", "content": "我妈妈上周做了手术", "turn_index": 0}],
            [],
        )

        assert len(result.nodes) == 2
        assert result.nodes[0].name == "mom"
        assert result.nodes[0].type == NodeType.PERSON
        assert len(result.edges) == 1
        assert result.edges[0].type == EdgeType.INVOLVED_IN

    @pytest.mark.asyncio
    async def test_extractor_handles_empty_response(self):
        async def mock_llm(system: str, user: str) -> str:
            return "I cannot extract any knowledge from this."

        extractor = Extractor(mock_llm)
        result = await extractor.extract([{"role": "user", "content": "hello", "turn_index": 0}], [])
        assert len(result.nodes) == 0
        assert len(result.edges) == 0

    @pytest.mark.asyncio
    async def test_extractor_filters_invalid_nodes(self):
        async def mock_llm(system: str, user: str) -> str:
            return json.dumps({
                "nodes": [
                    {"type": "INVALID_TYPE", "name": "bad", "content": "c"},
                    {"type": "PERSON", "name": "good", "content": "c"},  # missing description ok
                    {"type": "EVENT"},  # missing name and content
                ],
                "edges": [],
            })

        extractor = Extractor(mock_llm)
        result = await extractor.extract([{"role": "user", "content": "test", "turn_index": 0}], [])
        assert len(result.nodes) == 1
        assert result.nodes[0].name == "good"

    @pytest.mark.asyncio
    async def test_extractor_with_user_and_case(self):
        async def mock_llm(system: str, user: str) -> str:
            return json.dumps({
                "nodes": [
                    {"type": "USER", "name": "user-profile", "description": "user profile", "content": "user-profile\n基本信息: 25岁\n性格特点: 内向"},
                    {"type": "CASE", "name": "work-stress", "description": "work stress case", "content": "work-stress\n问题: 工作压力大\n原因: 项目截止期\n方案: 时间管理"},
                ],
                "edges": [],
            })

        extractor = Extractor(mock_llm)
        result = await extractor.extract([{"role": "user", "content": "我25岁，最近工作压力很大", "turn_index": 0}], [])

        assert len(result.nodes) == 2
        assert result.nodes[0].type == NodeType.USER
        assert result.nodes[1].type == NodeType.CASE


# ═════════════════════════════════════════════════════════════
#  11. Assemble — 上下文组装
# ═════════════════════════════════════════════════════════════


class TestAssemble:
    def test_build_system_prompt_empty(self):
        result = build_system_prompt_addition([], 0)
        assert result == ""

    def test_build_system_prompt_with_recalled(self):
        nodes = [
            {"type": "PERSON", "src": "recalled"},
            {"type": "EVENT", "src": "active"},
        ]
        result = build_system_prompt_addition(nodes, 2)
        assert "Graph Memory" in result
        assert "1 nodes recalled from OTHER" in result
        assert "1 people" in result

    def test_assemble_context_empty(self, db):
        result = assemble_context(db, [], [], [], [])
        assert result["xml"] is None
        assert result["tokens"] == 0

    def test_assemble_context_with_nodes(self, db):
        n1 = insert_node(db, name="person-1", type_="PERSON", content="person content", community_id="c-1")
        n2 = insert_node(db, name="event-1", type_="EVENT", content="event content", community_id="c-1")
        insert_edge(db, from_id=n1, to_id=n2, type_="INVOLVED_IN")

        node1 = store.find_by_id(db, n1)
        node2 = store.find_by_id(db, n2)
        edges = store.all_edges(db)

        result = assemble_context(db, [], [], [node1, node2], edges)
        assert result["xml"] is not None
        assert "<knowledge_graph>" in result["xml"]
        assert "person-1" in result["xml"]
        assert "event-1" in result["xml"]
        assert "INVOLVED_IN" in result["xml"]
        assert result["tokens"] > 0


# ═════════════════════════════════════════════════════════════
#  12. Recaller — 召回
# ═════════════════════════════════════════════════════════════


class TestRecaller:
    @pytest.mark.asyncio
    async def test_recall_empty_db(self, db, cfg):
        recaller = Recaller(db, cfg)
        result = await recaller.recall("test query")
        assert len(result.nodes) == 0
        assert len(result.edges) == 0

    @pytest.mark.asyncio
    async def test_recall_with_fts5_only(self, db, cfg):
        insert_node(db, name="mom", content="妈妈最近身体不太好需要多关心", validated_count=5)

        recaller = Recaller(db, cfg)
        result = await recaller.recall("妈妈")
        assert len(result.nodes) >= 1

    @pytest.mark.asyncio
    async def test_recall_with_vector(self, db, cfg):
        n1 = insert_node(db, name="mom", content="妈妈最近身体不太好")
        insert_vector(db, n1, [1.0, 0.0, 0.0])

        async def mock_embed(text: str) -> List[float]:
            return [1.0, 0.0, 0.0]

        recaller = Recaller(db, cfg)
        recaller.set_embed_fn(mock_embed)

        result = await recaller.recall("妈妈身体")
        assert len(result.nodes) >= 1
        assert result.nodes[0].name == "mom"

    @pytest.mark.asyncio
    async def test_sync_embed(self, db, cfg):
        n1 = insert_node(db, name="test-node", content="test content")

        async def mock_embed(text: str) -> List[float]:
            return [0.1, 0.2, 0.3]

        recaller = Recaller(db, cfg)
        recaller.set_embed_fn(mock_embed)

        node = store.find_by_id(db, n1)
        await recaller.sync_embed(node)

        h = store.get_vector_hash(db, n1)
        assert h is not None


# ═════════════════════════════════════════════════════════════
#  13. Maintenance — 完整维护流程
# ═════════════════════════════════════════════════════════════


class TestMaintenance:
    @pytest.mark.asyncio
    async def test_maintenance_empty_db(self, db, cfg):
        result = await run_maintenance(db, cfg)
        assert result.dedup.merged == 0
        assert result.community.count == 0
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_maintenance_with_data(self, db, cfg):
        n1 = insert_node(db, name="person-1", type_="PERSON", validated_count=5)
        n2 = insert_node(db, name="event-1", type_="EVENT", validated_count=3)
        n3 = insert_node(db, name="pattern-1", type_="PATTERN", validated_count=1)

        insert_edge(db, from_id=n1, to_id=n2, type_="INVOLVED_IN")
        insert_edge(db, from_id=n2, to_id=n3, type_="TRIGGERS")

        invalidate_graph_cache()
        result = await run_maintenance(db, cfg)

        assert result.community.count >= 1
        assert len(result.pagerank.scores) == 3

        # Verify pagerank was written to DB
        node = store.find_by_id(db, n1)
        assert node.pagerank > 0

    @pytest.mark.asyncio
    async def test_maintenance_with_dedup(self, db, cfg):
        n1 = insert_node(db, name="a", type_="TOPIC", validated_count=3)
        n2 = insert_node(db, name="b", type_="TOPIC", validated_count=1)

        insert_vector(db, n1, [1.0, 0.0, 0.0])
        insert_vector(db, n2, [0.99, 0.01, 0.0])

        invalidate_graph_cache()
        result = await run_maintenance(db, cfg)
        assert result.dedup.merged >= 1


# ═════════════════════════════════════════════════════════════
#  14. Backend — 集成测试
# ═════════════════════════════════════════════════════════════


class TestBackend:
    @pytest.mark.asyncio
    async def test_backend_name(self):
        backend = MemoryV3Backend()
        assert backend.name == "v3"

    @pytest.mark.asyncio
    async def test_backend_get_recent_memories(self):
        backend = MemoryV3Backend()
        # Use in-memory db
        db = get_db(":memory:")
        backend._dbs["test-char"] = db

        insert_node(db, name="mom", validated_count=5)
        insert_node(db, name="anime", validated_count=2)

        memories = await backend.get_recent_memories("test-char", 5)
        assert len(memories) == 2
        assert memories[0]["validated_count"] >= memories[1]["validated_count"]

        close_db(db)

    @pytest.mark.asyncio
    async def test_backend_get_graph_stats(self):
        backend = MemoryV3Backend()
        db = get_db(":memory:")
        backend._dbs["test-char"] = db

        insert_node(db, name="person-1", type_="PERSON")
        insert_node(db, name="topic-1", type_="TOPIC")

        stats = await backend.get_graph_stats("test-char")
        assert stats["totalNodes"] == 2
        assert stats["byType"]["PERSON"] == 1

        close_db(db)

    @pytest.mark.asyncio
    async def test_backend_search_empty(self):
        backend = MemoryV3Backend()
        db = get_db(":memory:")
        backend._dbs["test-char"] = db

        results = await backend.search("test query", "test-char", 5)
        assert len(results) == 1  # returns one result dict even if empty
        assert results[0]["xml"] is None

        close_db(db)

    @pytest.mark.asyncio
    async def test_backend_ingest_message(self):
        backend = MemoryV3Backend()
        db = get_db(":memory:")
        backend._dbs["test-char"] = db

        await backend.ingest_message("test-char", "sess-1", "user", "hello", turn_index=0)
        await backend.ingest_message("test-char", "sess-1", "assistant", "hi", turn_index=1)

        # Verify messages were saved
        msgs = store.get_unextracted(db, "sess-1", 10)
        assert len(msgs) == 2

        close_db(db)
