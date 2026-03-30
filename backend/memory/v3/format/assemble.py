"""
Graph Memory V3 — 上下文组装

构建 XML <knowledge_graph> + <episodic_context> 注入到 Agent 系统提示
"""

from __future__ import annotations

import sqlite3
from typing import Dict, List, Any
from xml.sax.saxutils import escape as _sax_escape

from ..types import GmEdge, GmNode
from ..store.store import get_community_summary, get_episodic_messages

CHARS_PER_TOKEN = 3


def _xml_escape(s: str) -> str:
    return _sax_escape(s)


def build_system_prompt_addition(
    selected_nodes: List[Dict[str, str]],
    edge_count: int,
) -> str:
    if not selected_nodes:
        return ""

    recalled_count = sum(1 for n in selected_nodes if n.get("src") == "recalled")
    has_recalled = recalled_count > 0
    user_count = sum(1 for n in selected_nodes if n.get("type") == "USER")
    person_count = sum(1 for n in selected_nodes if n.get("type") == "PERSON")
    topic_count = sum(1 for n in selected_nodes if n.get("type") == "TOPIC")
    event_count = sum(1 for n in selected_nodes if n.get("type") == "EVENT")
    pattern_count = sum(1 for n in selected_nodes if n.get("type") == "PATTERN")
    case_count = sum(1 for n in selected_nodes if n.get("type") == "CASE")
    preference_count = sum(1 for n in selected_nodes if n.get("type") == "PREFERENCE")
    is_rich = len(selected_nodes) >= 4 or edge_count >= 3

    sections = [
        "## Graph Memory — 知识图谱记忆",
        "",
        "Below `<knowledge_graph>` is your accumulated memory from past conversations.",
        "It contains structured knowledge about the user — profile, people, topics, events, patterns, cases, and preferences.",
        "",
        f"Current graph: {user_count} user profile, {person_count} people, {topic_count} topics, {event_count} events, {pattern_count} patterns, {case_count} cases, {preference_count} preferences, {edge_count} relationships.",
    ]

    if has_recalled:
        sections += [
            "",
            f"**{recalled_count} nodes recalled from OTHER conversations** — these are proven solutions that worked before.",
            "Apply them directly when the current situation matches their trigger conditions.",
        ]

    sections += [
        "",
        "## Recalled context for this query",
        "",
        "This is a context engine. The following was retrieved by semantic search for the current message:",
        "",
        "- **`<episodic_context>`** — Trimmed conversation traces from sessions that produced the knowledge nodes, ordered by time.",
        "- **`<knowledge_graph>`** — Relevant nodes (USER/PERSON/TOPIC/EVENT/PATTERN/CASE/PREFERENCE) and edges, grouped by community.",
        "",
        "Read this context first.",
    ]

    if is_rich:
        sections += [
            "",
            "**Graph navigation:** Edges show how memories connect:",
            "- `CARES_ABOUT`: someone cares about a person or topic — be mindful of their feelings",
            "- `INVOLVED_IN`: a person participated in an event — reference their role when relevant",
            "- `TRIGGERS`: a topic or event triggers an emotional pattern — be sensitive to these triggers",
            "- `LEADS_TO`: one event led to another — understand the causal chain of the user's life",
            "- `HAS_PREFERENCE`: the user holds a specific preference — always respect their stated preferences",
            "- `RESOLVED_BY`: a case/problem was resolved by an event or pattern — learn from past solutions",
            "- `RELATED_TO`: general association between any two nodes",
        ]

    return "\n".join(sections)


def assemble_context(
    db: sqlite3.Connection,
    active_nodes: List[GmNode],
    active_edges: List[GmEdge],
    recalled_nodes: List[GmNode],
    recalled_edges: List[GmEdge],
) -> Dict[str, Any]:
    """
    组装知识图谱为 XML context

    Returns:
        {
            "xml": str | None,
            "system_prompt": str,
            "tokens": int,
            "episodic_xml": str,
            "episodic_tokens": int,
        }
    """
    # 合并节点：recall 结果优先被 active 覆盖
    node_map: Dict[str, Dict[str, Any]] = {}
    for n in recalled_nodes:
        node_map[n.id] = {**_node_to_dict(n), "src": "recalled"}
    for n in active_nodes:
        node_map[n.id] = {**_node_to_dict(n), "src": "active"}

    TYPE_PRI = {"USER": 7, "PREFERENCE": 6, "CASE": 5, "PATTERN": 4, "PERSON": 3, "EVENT": 2, "TOPIC": 1}
    sorted_nodes = sorted(
        node_map.values(),
        key=lambda n: (
            0 if n["src"] == "active" else 1,
            -(TYPE_PRI.get(n["type"], 0)),
            -n["validated_count"],
            -n["pagerank"],
        ),
    )

    selected = [n for n in sorted_nodes if n["status"] == "active"]
    if not selected:
        return {"xml": None, "system_prompt": "", "tokens": 0, "episodic_xml": "", "episodic_tokens": 0}

    id_to_name = {n["id"]: n["name"] for n in selected}
    selected_ids = {n["id"] for n in selected}

    all_edges = []
    seen_edge_ids = set()
    for e in list(active_edges) + list(recalled_edges):
        if e.id not in seen_edge_ids and e.from_id in selected_ids and e.to_id in selected_ids:
            all_edges.append(e)
            seen_edge_ids.add(e.id)

    # 按社区分组
    by_community: Dict[str, List[Dict]] = {}
    no_community: List[Dict] = []
    for n in selected:
        cid = n.get("community_id")
        if cid:
            by_community.setdefault(cid, []).append(n)
        else:
            no_community.append(n)

    # 生成 XML
    xml_parts: List[str] = []

    for cid, members in by_community.items():
        summary = get_community_summary(db, cid)
        label = _xml_escape(summary.summary) if summary else cid
        xml_parts.append(f'  <community id="{cid}" desc="{label}">')
        for n in members:
            xml_parts.append(_node_to_xml(n))
        xml_parts.append("  </community>")

    for n in no_community:
        xml_parts.append(_node_to_xml(n))

    nodes_xml = "\n".join(xml_parts)

    edges_xml = ""
    if all_edges:
        edge_lines = []
        for e in all_edges:
            from_name = id_to_name.get(e.from_id, e.from_id)
            to_name = id_to_name.get(e.to_id, e.to_id)
            cond = f' when="{_xml_escape(e.condition)}"' if e.condition else ""
            edge_lines.append(f'    <e type="{e.type.value}" from="{from_name}" to="{to_name}"{cond}>{_xml_escape(e.instruction)}</e>')
        edges_xml = "\n  <edges>\n" + "\n".join(edge_lines) + "\n  </edges>"

    xml = f"<knowledge_graph>\n{nodes_xml}{edges_xml}\n</knowledge_graph>"

    system_prompt = build_system_prompt_addition(
        selected_nodes=[{"type": n["type"], "src": n["src"]} for n in selected],
        edge_count=len(all_edges),
    )

    # 溯源选拉：top 3 节点的原始对话
    top_nodes = selected[:3]
    episodic_parts: List[str] = []

    for n in top_nodes:
        sessions = n.get("source_sessions", [])
        if not sessions:
            continue
        recent_sessions = sessions[-2:]
        msgs = get_episodic_messages(db, recent_sessions, n["updated_at"], 500)
        if not msgs:
            continue

        lines = [f'    [{m["role"].upper()}] {_xml_escape(m["text"][:200])}' for m in msgs]
        episodic_parts.append(f'  <trace node="{n["name"]}">\n' + "\n".join(lines) + f"\n  </trace>")

    episodic_xml = ""
    if episodic_parts:
        episodic_xml = "<episodic_context>\n" + "\n".join(episodic_parts) + "\n</episodic_context>"

    full_content = system_prompt + "\n\n" + xml + ("\n\n" + episodic_xml if episodic_xml else "")
    return {
        "xml": xml,
        "system_prompt": system_prompt,
        "tokens": len(full_content) // CHARS_PER_TOKEN,
        "episodic_xml": episodic_xml,
        "episodic_tokens": len(episodic_xml) // CHARS_PER_TOKEN,
    }


def _node_to_dict(n: GmNode) -> Dict[str, Any]:
    return {
        "id": n.id,
        "type": n.type.value,
        "name": n.name,
        "description": n.description,
        "content": n.content,
        "status": n.status.value,
        "validated_count": n.validated_count,
        "source_sessions": n.source_sessions,
        "community_id": n.community_id,
        "pagerank": n.pagerank,
        "updated_at": n.updated_at,
    }


def _node_to_xml(n: Dict[str, Any]) -> str:
    tag = n["type"].lower()
    src_attr = ' source="recalled"' if n.get("src") == "recalled" else ""
    from datetime import datetime
    time_attr = f' updated="{datetime.fromtimestamp(n["updated_at"] / 1000).strftime("%Y-%m-%d")}"'
    return (
        f'    <{tag} name="{n["name"]}" desc="{_xml_escape(n["description"])}"{src_attr}{time_attr}>\n'
        f'{n["content"].strip()}\n'
        f"    </{tag}>"
    )
