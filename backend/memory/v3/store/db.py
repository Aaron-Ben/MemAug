"""
Graph Memory V3 — SQLite 数据库初始化与迁移

8 个迁移：
  m1: 核心表（节点 + 边）
  m2: 消息存储
  m3: 信号存储
  m4: FTS5 全文索引
  m5: 向量存储
  m6: 社区描述存储
  m7: 情感陪伴类型体系（节点类型 PERSON/TOPIC/EVENT/PATTERN/PREFERENCE + 边类型 CARES_ABOUT/INVOLVED_IN/TRIGGERS/LEADS_TO/RELATED_TO）
  m8: USER/CASE 节点 + HAS_PREFERENCE/RESOLVED_BY 边
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_db(db_path: str) -> sqlite3.Connection:
    """获取或创建 SQLite 数据库连接"""
    resolved = _resolve_path(db_path)
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(resolved, check_same_thread=False)
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA foreign_keys = ON")
    _migrate(db)
    return db


def _resolve_path(p: str) -> str:
    if p.startswith("~"):
        return os.path.expanduser(p)
    return p


def close_db(db: Optional[sqlite3.Connection]) -> None:
    if db:
        try:
            db.close()
        except Exception:
            pass


def _migrate(db: sqlite3.Connection) -> None:
    db.execute(
        "CREATE TABLE IF NOT EXISTS _migrations (v INTEGER PRIMARY KEY, at INTEGER NOT NULL)"
    )
    row = db.execute("SELECT MAX(v) as v FROM _migrations").fetchone()
    cur = row[0] if row[0] is not None else 0

    steps = [_m1_core, _m2_messages, _m3_signals, _m4_fts5, _m5_vectors, _m6_communities, _m7_emotional_types, _m8_user_case_types]
    for i in range(cur, len(steps)):
        steps[i](db)
        db.execute("INSERT INTO _migrations (v, at) VALUES (?, ?)", (i + 1, int(time.time() * 1000)))
        db.commit()
        logger.info(f"[graph-memory] Migration {i + 1} applied")


# ─── 核心表：节点 + 边 ──────────────────────────────────────


def _m1_core(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS gm_nodes (
            id              TEXT PRIMARY KEY,
            type            TEXT NOT NULL CHECK(type IN ('TASK','SKILL','EVENT')),
            name            TEXT NOT NULL,
            description     TEXT NOT NULL DEFAULT '',
            content         TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','deprecated')),
            validated_count INTEGER NOT NULL DEFAULT 1,
            source_sessions TEXT NOT NULL DEFAULT '[]',
            community_id    TEXT,
            pagerank        REAL NOT NULL DEFAULT 0,
            created_at      INTEGER NOT NULL,
            updated_at      INTEGER NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_gm_nodes_name ON gm_nodes(name);
        CREATE INDEX IF NOT EXISTS ix_gm_nodes_type_status ON gm_nodes(type, status);
        CREATE INDEX IF NOT EXISTS ix_gm_nodes_community ON gm_nodes(community_id);

        CREATE TABLE IF NOT EXISTS gm_edges (
            id          TEXT PRIMARY KEY,
            from_id     TEXT NOT NULL REFERENCES gm_nodes(id),
            to_id       TEXT NOT NULL REFERENCES gm_nodes(id),
            type        TEXT NOT NULL CHECK(type IN ('USED_SKILL','SOLVED_BY','REQUIRES','PATCHES','CONFLICTS_WITH')),
            instruction TEXT NOT NULL,
            condition   TEXT,
            session_id  TEXT NOT NULL,
            created_at  INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_gm_edges_from ON gm_edges(from_id);
        CREATE INDEX IF NOT EXISTS ix_gm_edges_to   ON gm_edges(to_id);
    """)


# ─── 消息存储 ────────────────────────────────────────────────


def _m2_messages(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS gm_messages (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            turn_index  INTEGER NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            extracted   INTEGER NOT NULL DEFAULT 0,
            created_at  INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_gm_msg_session ON gm_messages(session_id, turn_index);
    """)


# ─── 信号存储 ────────────────────────────────────────────────


def _m3_signals(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS gm_signals (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            turn_index  INTEGER NOT NULL,
            type        TEXT NOT NULL,
            data        TEXT NOT NULL DEFAULT '{}',
            processed   INTEGER NOT NULL DEFAULT 0,
            created_at  INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_gm_sig_session ON gm_signals(session_id, processed);
    """)


# ─── FTS5 全文索引 ───────────────────────────────────────────


def _m4_fts5(db: sqlite3.Connection) -> None:
    try:
        db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS gm_nodes_fts USING fts5(
                name,
                description,
                content,
                content=gm_nodes,
                content_rowid=rowid
            );
        """)
        db.executescript("""
            CREATE TRIGGER IF NOT EXISTS gm_nodes_ai AFTER INSERT ON gm_nodes BEGIN
                INSERT INTO gm_nodes_fts(rowid, name, description, content)
                VALUES (NEW.rowid, NEW.name, NEW.description, NEW.content);
            END;
            CREATE TRIGGER IF NOT EXISTS gm_nodes_ad AFTER DELETE ON gm_nodes BEGIN
                INSERT INTO gm_nodes_fts(gm_nodes_fts, rowid, name, description, content)
                VALUES ('delete', OLD.rowid, OLD.name, OLD.description, OLD.content);
            END;
            CREATE TRIGGER IF NOT EXISTS gm_nodes_au AFTER UPDATE ON gm_nodes BEGIN
                INSERT INTO gm_nodes_fts(gm_nodes_fts, rowid, name, description, content)
                VALUES ('delete', OLD.rowid, OLD.name, OLD.description, OLD.content);
                INSERT INTO gm_nodes_fts(rowid, name, description, content)
                VALUES (NEW.rowid, NEW.name, NEW.description, NEW.content);
            END;
        """)
    except sqlite3.OperationalError:
        # FTS5 不可用时静默降级到 LIKE 搜索
        logger.warning("[graph-memory] FTS5 not available, falling back to LIKE search")


# ─── 向量存储 ────────────────────────────────────────────────


def _m5_vectors(db: sqlite3.Connection) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS gm_vectors (
            node_id      TEXT PRIMARY KEY REFERENCES gm_nodes(id),
            content_hash TEXT NOT NULL,
            embedding    BLOB NOT NULL
        );
    """)


# ─── 社区描述存储 ────────────────────────────────────────────


def _m6_communities(db: sqlite3.Connection) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS gm_communities (
            id          TEXT PRIMARY KEY,
            summary     TEXT NOT NULL,
            node_count  INTEGER NOT NULL DEFAULT 0,
            embedding   BLOB,
            created_at  INTEGER NOT NULL,
            updated_at  INTEGER NOT NULL
        );
    """)


# ─── 情感陪伴类型体系 ─────────────────────────────────────────


def _m7_emotional_types(db: sqlite3.Connection) -> None:
    """将节点/边类型从开发助手体系迁移到情感陪伴体系。"""
    # ── 重建 gm_nodes（新 CHECK 约束）──
    db.executescript("""
        CREATE TABLE gm_nodes_new (
            id              TEXT PRIMARY KEY,
            type            TEXT NOT NULL CHECK(type IN ('PERSON','TOPIC','EVENT','PATTERN','PREFERENCE')),
            name            TEXT NOT NULL,
            description     TEXT NOT NULL DEFAULT '',
            content         TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','deprecated')),
            validated_count INTEGER NOT NULL DEFAULT 1,
            source_sessions TEXT NOT NULL DEFAULT '[]',
            community_id    TEXT,
            pagerank        REAL NOT NULL DEFAULT 0,
            created_at      INTEGER NOT NULL,
            updated_at      INTEGER NOT NULL
        );
    """)
    rows = db.execute("SELECT * FROM gm_nodes").fetchall()
    cols = [d[0] for d in db.execute("SELECT * FROM gm_nodes LIMIT 0").description]
    if rows:
        placeholders = ",".join(["?"] * len(cols))
        db.executemany(
            f"INSERT INTO gm_nodes_new ({','.join(cols)}) VALUES ({placeholders})",
            rows,
        )
    db.executescript("DROP TABLE gm_nodes; ALTER TABLE gm_nodes_new RENAME TO gm_nodes;")
    db.executescript("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_gm_nodes_name ON gm_nodes(name);
        CREATE INDEX IF NOT EXISTS ix_gm_nodes_type_status ON gm_nodes(type, status);
        CREATE INDEX IF NOT EXISTS ix_gm_nodes_community ON gm_nodes(community_id);
    """)

    # ── 重建 gm_edges（新 CHECK 约束）──
    db.executescript("""
        CREATE TABLE gm_edges_new (
            id          TEXT PRIMARY KEY,
            from_id     TEXT NOT NULL REFERENCES gm_nodes(id),
            to_id       TEXT NOT NULL REFERENCES gm_nodes(id),
            type        TEXT NOT NULL CHECK(type IN ('CARES_ABOUT','INVOLVED_IN','TRIGGERS','LEADS_TO','RELATED_TO')),
            instruction TEXT NOT NULL,
            condition   TEXT,
            session_id  TEXT NOT NULL,
            created_at  INTEGER NOT NULL
        );
    """)
    # 旧边无法自动映射到新类型，直接丢弃
    db.executescript("DROP TABLE gm_edges; ALTER TABLE gm_edges_new RENAME TO gm_edges;")
    db.executescript("""
        CREATE INDEX IF NOT EXISTS ix_gm_edges_from ON gm_edges(from_id);
        CREATE INDEX IF NOT EXISTS ix_gm_edges_to   ON gm_edges(to_id);
    """)

    # ── 重建 FTS5（依赖 gm_nodes）──
    try:
        db.executescript("DROP TABLE IF EXISTS gm_nodes_fts;")
        db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS gm_nodes_fts USING fts5(
                name,
                description,
                content,
                content=gm_nodes,
                content_rowid=rowid
            );
        """)
        db.executescript("""
            CREATE TRIGGER IF NOT EXISTS gm_nodes_ai AFTER INSERT ON gm_nodes BEGIN
                INSERT INTO gm_nodes_fts(rowid, name, description, content)
                VALUES (NEW.rowid, NEW.name, NEW.description, NEW.content);
            END;
            CREATE TRIGGER IF NOT EXISTS gm_nodes_ad AFTER DELETE ON gm_nodes BEGIN
                INSERT INTO gm_nodes_fts(gm_nodes_fts, rowid, name, description, content)
                VALUES ('delete', OLD.rowid, OLD.name, OLD.description, OLD.content);
            END;
            CREATE TRIGGER IF NOT EXISTS gm_nodes_au AFTER UPDATE ON gm_nodes BEGIN
                INSERT INTO gm_nodes_fts(gm_nodes_fts, rowid, name, description, content)
                VALUES ('delete', OLD.rowid, OLD.name, OLD.description, OLD.content);
                INSERT INTO gm_nodes_fts(rowid, name, description, content)
                VALUES (NEW.rowid, NEW.name, NEW.description, NEW.content);
            END;
        """)
    except sqlite3.OperationalError:
        logger.warning("[graph-memory] FTS5 rebuild failed during m7, skipping")


# ─── USER/CASE 节点 + HAS_PREFERENCE/RESOLVED_BY 边 ───────────


def _m8_user_case_types(db: sqlite3.Connection) -> None:
    """增加 USER/CASE 节点类型和 HAS_PREFERENCE/RESOLVED_BY 边类型。"""
    # ── 重建 gm_nodes（扩展 CHECK 约束）──
    db.executescript("""
        CREATE TABLE gm_nodes_new (
            id              TEXT PRIMARY KEY,
            type            TEXT NOT NULL CHECK(type IN ('USER','PERSON','TOPIC','EVENT','PATTERN','CASE','PREFERENCE')),
            name            TEXT NOT NULL,
            description     TEXT NOT NULL DEFAULT '',
            content         TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','deprecated')),
            validated_count INTEGER NOT NULL DEFAULT 1,
            source_sessions TEXT NOT NULL DEFAULT '[]',
            community_id    TEXT,
            pagerank        REAL NOT NULL DEFAULT 0,
            created_at      INTEGER NOT NULL,
            updated_at      INTEGER NOT NULL
        );
    """)
    rows = db.execute("SELECT * FROM gm_nodes").fetchall()
    cols = [d[0] for d in db.execute("SELECT * FROM gm_nodes LIMIT 0").description]
    if rows:
        placeholders = ",".join(["?"] * len(cols))
        db.executemany(
            f"INSERT INTO gm_nodes_new ({','.join(cols)}) VALUES ({placeholders})",
            rows,
        )
    db.executescript("DROP TABLE gm_nodes; ALTER TABLE gm_nodes_new RENAME TO gm_nodes;")
    db.executescript("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_gm_nodes_name ON gm_nodes(name);
        CREATE INDEX IF NOT EXISTS ix_gm_nodes_type_status ON gm_nodes(type, status);
        CREATE INDEX IF NOT EXISTS ix_gm_nodes_community ON gm_nodes(community_id);
    """)

    # ── 重建 gm_edges（扩展 CHECK 约束）──
    db.executescript("""
        CREATE TABLE gm_edges_new (
            id          TEXT PRIMARY KEY,
            from_id     TEXT NOT NULL REFERENCES gm_nodes(id),
            to_id       TEXT NOT NULL REFERENCES gm_nodes(id),
            type        TEXT NOT NULL CHECK(type IN ('CARES_ABOUT','INVOLVED_IN','TRIGGERS','LEADS_TO','HAS_PREFERENCE','RESOLVED_BY','RELATED_TO')),
            instruction TEXT NOT NULL,
            condition   TEXT,
            session_id  TEXT NOT NULL,
            created_at  INTEGER NOT NULL
        );
    """)
    rows = db.execute("SELECT * FROM gm_edges").fetchall()
    cols = [d[0] for d in db.execute("SELECT * FROM gm_edges LIMIT 0").description]
    if rows:
        placeholders = ",".join(["?"] * len(cols))
        db.executemany(
            f"INSERT INTO gm_edges_new ({','.join(cols)}) VALUES ({placeholders})",
            rows,
        )
    db.executescript("DROP TABLE gm_edges; ALTER TABLE gm_edges_new RENAME TO gm_edges;")
    db.executescript("""
        CREATE INDEX IF NOT EXISTS ix_gm_edges_from ON gm_edges(from_id);
        CREATE INDEX IF NOT EXISTS ix_gm_edges_to   ON gm_edges(to_id);
    """)

    # ── 重建 FTS5 ──
    try:
        db.executescript("DROP TABLE IF EXISTS gm_nodes_fts;")
        db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS gm_nodes_fts USING fts5(
                name,
                description,
                content,
                content=gm_nodes,
                content_rowid=rowid
            );
        """)
        db.executescript("""
            CREATE TRIGGER IF NOT EXISTS gm_nodes_ai AFTER INSERT ON gm_nodes BEGIN
                INSERT INTO gm_nodes_fts(rowid, name, description, content)
                VALUES (NEW.rowid, NEW.name, NEW.description, NEW.content);
            END;
            CREATE TRIGGER IF NOT EXISTS gm_nodes_ad AFTER DELETE ON gm_nodes BEGIN
                INSERT INTO gm_nodes_fts(gm_nodes_fts, rowid, name, description, content)
                VALUES ('delete', OLD.rowid, OLD.name, OLD.description, OLD.content);
            END;
            CREATE TRIGGER IF NOT EXISTS gm_nodes_au AFTER UPDATE ON gm_nodes BEGIN
                INSERT INTO gm_nodes_fts(gm_nodes_fts, rowid, name, description, content)
                VALUES ('delete', OLD.rowid, OLD.name, OLD.description, OLD.content);
                INSERT INTO gm_nodes_fts(rowid, name, description, content)
                VALUES (NEW.rowid, NEW.name, NEW.description, NEW.content);
            END;
        """)
    except sqlite3.OperationalError:
        logger.warning("[graph-memory] FTS5 rebuild failed during m8, skipping")
