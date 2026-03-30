"""
Graph Memory V3 — 知识提取引擎

通过 LLM 从情感陪伴对话中提取结构化知识（节点 + 关系）
以及 session 结束前的 finalize（EVENT→PATTERN 升级、补充关系、标记失效）
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from ..types import EdgeCandidate, EdgeType, ExtractionResult, FinalizeResult, NodeCandidate, NodeType, PromotedPattern
from ..store.store import normalize_name

logger = logging.getLogger(__name__)

# ─── LLM 函数签名 ────────────────────────────────────────────

CompleteFn = Callable[[str, str], Coroutine[Any, Any, str]]


# ─── 节点/边合法值 ──────────────────────────────────────────

VALID_NODE_TYPES: Set[str] = {"USER", "PERSON", "TOPIC", "EVENT", "PATTERN", "CASE", "PREFERENCE"}
VALID_EDGE_TYPES: Set[str] = {"CARES_ABOUT", "INVOLVED_IN", "TRIGGERS", "LEADS_TO", "HAS_PREFERENCE", "RESOLVED_BY", "RELATED_TO"}

EDGE_FROM_CONSTRAINT: Dict[str, Set[str]] = {
    "CARES_ABOUT": {"USER", "PERSON", "TOPIC", "EVENT", "PATTERN", "PREFERENCE"},
    "INVOLVED_IN": {"PERSON"},
    "TRIGGERS": {"TOPIC", "EVENT"},
    "LEADS_TO": {"EVENT"},
    "HAS_PREFERENCE": {"USER"},
    "RESOLVED_BY": {"CASE"},
    "RELATED_TO": {"USER", "PERSON", "TOPIC", "EVENT", "PATTERN", "CASE", "PREFERENCE"},
}

EDGE_TO_CONSTRAINT: Dict[str, Set[str]] = {
    "CARES_ABOUT": {"PERSON", "TOPIC"},
    "INVOLVED_IN": {"EVENT"},
    "TRIGGERS": {"PATTERN"},
    "LEADS_TO": {"EVENT"},
    "HAS_PREFERENCE": {"PREFERENCE"},
    "RESOLVED_BY": {"EVENT", "PATTERN"},
    "RELATED_TO": {"USER", "PERSON", "TOPIC", "EVENT", "PATTERN", "CASE", "PREFERENCE"},
}

# ─── 提取 System Prompt ─────────────────────────────────────

EXTRACT_SYS = """你是情感陪伴 AI 的记忆提取引擎，从用户与 AI 的对话中提取结构化记忆（节点 + 关系）。
提取的记忆将在未来对话中被召回，帮助 AI 记住用户的人、事、情感和偏好。
输出严格 JSON：{"nodes":[...],"edges":[...]}，不包含任何额外文字。

1. 节点提取：
   1.1 从对话中识别七类记忆节点：
       - USER：用户自身的基本信息，如年龄、性格、生活状态、重要背景。每个用户只有一个 USER 节点
       - PERSON：用户提到的重要人物，包括家人、朋友、同事、宠物等
       - TOPIC：兴趣爱好、常聊话题、反复提及的领域
       - EVENT：用户经历或即将经历的重要生活事件，包括里程碑、变化、转折
       - PATTERN：用户的情感或行为模式，如情绪触发因素、应对方式、反复出现的反应
       - CASE：用户遇到的问题及其解决过程，包含问题、原因、方案、结果
       - PREFERENCE：用户对 AI 回应方式的偏好，如沟通风格、安慰方式、话题边界
   1.2 每个节点必须包含 4 个字段，缺一不可：
       - type：节点类型，只允许 USER / PERSON / TOPIC / EVENT / PATTERN / CASE / PREFERENCE
       - name：全小写连字符命名，确保整个提取过程命名一致
       - description：一句话说明什么时候会用到这条记忆
       - content：纯文本格式的记忆内容（见 1.4 的模板）
   1.3 name 命名规范：
       - USER：固定名称，如 user-profile、me
       - PERSON：名字或称呼，如 mom、xiaoming、teacher-wang、cat-mimi
       - TOPIC：名词，如 anime、hiking、programming、cooking
       - EVENT：描述性短语，如 graduation-trip、move-to-shanghai、first-job
       - PATTERN：现象描述，如 exam-anxiety、late-night-emo、stress-eating
       - CASE：问题描述，如 insomnia-case、work-burnout、relationship-conflict
       - PREFERENCE：偏好描述，如 prefer-encouragement、dislike-lecturing、like-detailed-reply
       - 已有节点列表会提供，相同事物必须复用已有 name，不得创建重复节点
   1.4 content 模板（纯文本，按 type 选用）：
       USER → "[name]\\n基本信息: ...\\n性格特点: ...\\n生活状态: ...\\n重要背景: ..."
       PERSON → "[name]\\n关系: ...\\n描述: ...\\n特点: ...\\n互动模式: ..."
       TOPIC → "[name]\\n类型: ...\\n描述: ...\\n相关经历: ..."
       EVENT → "[name]\\n时间: ...\\n经过: ...\\n情感影响: ...\\n后续: ..."
       PATTERN → "[name]\\n触发条件: ...\\n表现形式: ...\\n应对方式: ..."
       CASE → "[name]\\n问题: ...\\n原因: ...\\n方案: ...\\n结果: ..."
       PREFERENCE → "[name]\\n偏好内容: ...\\n背景原因: ...\\n表现方式: ..."

2. 关系提取：
   2.1 识别节点之间直接、明确的关系，只允许以下 7 种边类型。
   2.2 每条边必须包含 from、to、type、instruction 四个字段，缺一不可。
   2.3 边类型定义与方向约束（严格遵守，不得混用）：

       CARES_ABOUT
         方向：USER/PERSON/TOPIC/EVENT/PATTERN/PREFERENCE → PERSON 或 TOPIC（且仅限此方向）
         含义：某人或某事关心/在意另一个人或某个话题
         instruction：写具体关心的内容、情感色彩、关注程度
         判定：to 节点是 PERSON 或 TOPIC

       INVOLVED_IN
         方向：PERSON → EVENT（且仅限此方向）
         含义：某个人参与/经历了某个事件
         instruction：写该人在事件中的角色、参与方式
         判定：from 节点是 PERSON，to 节点是 EVENT

       TRIGGERS
         方向：TOPIC 或 EVENT → PATTERN（且仅限此方向）
         含义：某个话题或事件触发了某种情感/行为模式
         instruction：写触发机制、从什么到什么的转变
         condition（必填）：写什么具体条件触发了这个模式
         判定：from 节点是 TOPIC 或 EVENT，to 节点是 PATTERN

       LEADS_TO
         方向：EVENT → EVENT（且仅限此方向）
         含义：一个事件导致了另一个事件（因果或时序）
         instruction：写因果链条、时间间隔、影响程度

       HAS_PREFERENCE
         方向：USER → PREFERENCE（且仅限此方向）
         含义：用户持有某种偏好
         instruction：写具体偏好内容、偏好程度、形成背景
         判定：from 节点是 USER，to 节点是 PREFERENCE

       RESOLVED_BY
         方向：CASE → EVENT 或 PATTERN（且仅限此方向）
         含义：某个问题/案例通过某个事件或模式得到了解决
         instruction：写解决机制、有效性、最终结果
         判定：from 节点是 CASE，to 节点是 EVENT 或 PATTERN

       RELATED_TO
         方向：任意类型 → 任意类型（双向）
         含义：两个节点有关联但无法归入以上六种类型
         instruction：写关联的具体内容和关联强度

   2.4 关系方向选择决策树（按此顺序判定）：
       a. from 是 USER，to 是 PREFERENCE → 必须用 HAS_PREFERENCE
       b. from 是 CASE，to 是 EVENT 或 PATTERN → 必须用 RESOLVED_BY
       c. from 是 PERSON，to 是 EVENT → 必须用 INVOLVED_IN
       d. from 是 TOPIC/EVENT，to 是 PATTERN → 必须用 TRIGGERS
       e. from 是 EVENT，to 是 EVENT → 必须用 LEADS_TO
       f. to 是 PERSON 或 TOPIC → 用 CARES_ABOUT
       g. 以上均不满足 → 用 RELATED_TO
       h. 只有存在直接、明确的关系时才提取边，不要强行关联

3. 提取策略（宁多勿漏）：
   3.1 用户的自我介绍、基本信息、性格描述，提取为 USER
   3.2 用户提到的人名、地点、机构等实体，都应提取为 PERSON 或 TOPIC
   3.3 用户分享的情感体验、情绪变化，提取为 EVENT 或 PATTERN
   3.4 用户遇到的问题及解决过程，提取为 CASE
   3.5 用户对 AI 表达的偏好或不满，提取为 PREFERENCE
   3.6 只有纯粹的寒暄问候（如"你好""谢谢"）才不提取
   3.7 重复提及同一人或同一话题时，必须复用已有节点 name

4. 输出规范：
   4.1 只返回 JSON，格式为 {"nodes":[...],"edges":[...]}
   4.2 禁止 markdown 代码块包裹，禁止解释文字，禁止额外字段
   4.3 没有值得记忆的内容时返回 {"nodes":[],"edges":[]}
   4.4 每条 edge 的 instruction 必须写具体内容，不能为空或写"见上文"
"""


# ─── 整理 System Prompt ─────────────────────────────────────

FINALIZE_SYS = """你是图谱节点整理引擎，对本次对话产生的记忆节点做 session 结束前的最终审查。
审查本次对话所有节点，执行以下三项操作，输出严格 JSON。

1. EVENT 升级为 PATTERN：
   如果某个 EVENT 节点反映了反复出现的情感或行为模式（不限于单次经历），将其升级为 PATTERN。
   升级时需要：改名为 PATTERN 命名规范（现象描述）、完善 content 为 PATTERN 模板格式（触发条件/表现形式/应对方式）。
   写入 promotedPatterns 数组。

2. 补充遗漏关系：
   整体回顾所有节点，发现单次提取时难以察觉的跨节点关系。
   关系类型只允许：CARES_ABOUT、INVOLVED_IN、TRIGGERS、LEADS_TO、HAS_PREFERENCE、RESOLVED_BY、RELATED_TO。
   严格遵守方向约束：USER→PREFERENCE 用 HAS_PREFERENCE，CASE→EVENT/PATTERN 用 RESOLVED_BY，PERSON→EVENT 用 INVOLVED_IN，TOPIC/EVENT→PATTERN 用 TRIGGERS，EVENT→EVENT 用 LEADS_TO。
   写入 newEdges 数组。

3. 标记失效节点：
   因本次对话中的新发现而过时或不再准确的旧节点，将其 node_id 写入 invalidations 数组。

没有需要处理的项返回空数组。只返回 JSON，禁止额外文字。
格式：{"promotedPatterns":[{"type":"PATTERN","name":"...","description":"...","content":"..."}],"newEdges":[{"from":"...","to":"...","type":"...","instruction":"..."}],"invalidations":["node-id"]}"""


# ─── 边类型自动修正 ─────────────────────────────────────────


def _correct_edge_type(
    edge: Dict[str, Any],
    name_to_type: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    from_name = normalize_name(edge.get("from", ""))
    to_name = normalize_name(edge.get("to", ""))
    from_type = name_to_type.get(from_name)
    to_type = name_to_type.get(to_name)

    if not from_type or not to_type:
        return edge

    etype = edge.get("type", "")

    # 自动修正：PERSON → EVENT 必须是 INVOLVED_IN
    if from_type == "PERSON" and to_type == "EVENT" and etype != "INVOLVED_IN":
        logger.debug(f"edge corrected: {from_name} ->[{etype}]-> {to_name} => INVOLVED_IN")
        etype = "INVOLVED_IN"

    # 自动修正：TOPIC/EVENT → PATTERN 必须是 TRIGGERS
    if from_type in ("TOPIC", "EVENT") and to_type == "PATTERN" and etype != "TRIGGERS":
        logger.debug(f"edge corrected: {from_name} ->[{etype}]-> {to_name} => TRIGGERS")
        etype = "TRIGGERS"

    # 自动修正：EVENT → EVENT 必须是 LEADS_TO
    if from_type == "EVENT" and to_type == "EVENT" and etype != "LEADS_TO":
        logger.debug(f"edge corrected: {from_name} ->[{etype}]-> {to_name} => LEADS_TO")
        etype = "LEADS_TO"

    # 自动修正：USER → PREFERENCE 必须是 HAS_PREFERENCE
    if from_type == "USER" and to_type == "PREFERENCE" and etype != "HAS_PREFERENCE":
        logger.debug(f"edge corrected: {from_name} ->[{etype}]-> {to_name} => HAS_PREFERENCE")
        etype = "HAS_PREFERENCE"

    # 自动修正：CASE → EVENT/PATTERN 必须是 RESOLVED_BY
    if from_type == "CASE" and to_type in ("EVENT", "PATTERN") and etype != "RESOLVED_BY":
        logger.debug(f"edge corrected: {from_name} ->[{etype}]-> {to_name} => RESOLVED_BY")
        etype = "RESOLVED_BY"

    if etype not in VALID_EDGE_TYPES:
        logger.debug(f"edge dropped: invalid type '{etype}'")
        return None

    from_ok = from_type in EDGE_FROM_CONSTRAINT.get(etype, set())
    to_ok = to_type in EDGE_TO_CONSTRAINT.get(etype, set())
    if not from_ok or not to_ok:
        logger.debug(f"edge dropped: {from_type}->[{etype}]->{to_type} violates direction constraint")
        return None

    edge = dict(edge)
    edge["type"] = etype
    return edge


# ─── JSON 提取 ───────────────────────────────────────────────


def _extract_json(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"<think[\s\S]*?<\/think>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<think[\s\S]*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^```(?:json)?\s*\n?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n?\s*```\s*$", "", s, flags=re.IGNORECASE)
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    if s.startswith("[") and s.endswith("]"):
        return s
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last > first:
        return s[first:last + 1]
    return s


# ─── Extractor ───────────────────────────────────────────────


class Extractor:
    def __init__(self, llm: CompleteFn):
        self._llm = llm

    async def extract(
        self,
        messages: List[Dict[str, Any]],
        existing_names: List[str],
    ) -> ExtractionResult:
        msgs_text = "\n\n---\n\n".join(
            f"[{(m.get('role', '?')).upper()} t={m.get('turn_index', 0)}]\n"
            + str(m.get("content", ""))[:800]
            for m in messages
        )

        user_prompt = (
            f"<Existing Nodes>\n{', '.join(existing_names) if existing_names else '（无）'}\n\n"
            f"<Conversation>\n{msgs_text}"
        )

        raw = await self._llm(EXTRACT_SYS, user_prompt)
        return self._parse_extract(raw)

    async def finalize(
        self,
        session_nodes: List[Dict[str, Any]],
        graph_summary: str,
    ) -> FinalizeResult:
        nodes_json = json.dumps(
            [
                {"id": n.get("id"), "type": n.get("type"), "name": n.get("name"),
                 "description": n.get("description"), "v": n.get("validated_count")}
                for n in session_nodes
            ],
            ensure_ascii=False,
            indent=2,
        )
        user_prompt = f"<Session Nodes>\n{nodes_json}\n\n<Graph Summary>\n{graph_summary}"
        raw = await self._llm(FINALIZE_SYS, user_prompt)
        return self._parse_finalize(raw, session_nodes)

    def _parse_extract(self, raw: str) -> ExtractionResult:
        try:
            json_str = _extract_json(raw)
            p = json.loads(json_str)

            nodes: List[NodeCandidate] = []
            for n in p.get("nodes", []):
                if not n.get("name") or not n.get("type") or not n.get("content"):
                    continue
                if n["type"] not in VALID_NODE_TYPES:
                    logger.debug(f"node dropped: invalid type '{n['type']}'")
                    continue
                nodes.append(NodeCandidate(
                    type=NodeType(n["type"]),
                    name=normalize_name(n["name"]),
                    description=n.get("description", ""),
                    content=n["content"],
                ))

            name_to_type = {nc.name: nc.type.value for nc in nodes}

            edges: List[EdgeCandidate] = []
            for e in p.get("edges", []):
                if not e.get("from") or not e.get("to") or not e.get("type") or not e.get("instruction"):
                    continue
                e["from"] = normalize_name(e["from"])
                e["to"] = normalize_name(e["to"])
                corrected = _correct_edge_type(e, name_to_type)
                if corrected is None:
                    continue
                edges.append(EdgeCandidate(
                    from_name=corrected["from"],
                    to_name=corrected["to"],
                    type=EdgeType(corrected["type"]),
                    instruction=corrected["instruction"],
                    condition=corrected.get("condition"),
                ))

            return ExtractionResult(nodes=nodes, edges=edges)
        except (json.JSONDecodeError, Exception) as err:
            logger.debug(f"JSON parse failed: {err}")
            return ExtractionResult(nodes=[], edges=[])

    def _parse_finalize(self, raw: str, session_nodes: Optional[List[Dict]] = None) -> FinalizeResult:
        try:
            json_str = _extract_json(raw)
            p = json.loads(json_str)

            name_to_type: Dict[str, str] = {}
            if session_nodes:
                for n in session_nodes:
                    if n.get("name") and n.get("type"):
                        name_to_type[normalize_name(n["name"])] = n["type"]

            promoted_patterns = []
            for n in p.get("promotedPatterns", []):
                if n.get("name") and n.get("content"):
                    name_to_type[normalize_name(n["name"])] = n.get("type", "PATTERN")
                    promoted_patterns.append(PromotedPattern(
                        name=normalize_name(n["name"]),
                        description=n.get("description", ""),
                        content=n["content"],
                    ))

            new_edges = []
            for e in p.get("newEdges", []):
                if not e.get("from") or not e.get("to") or not e.get("type") or e["type"] not in VALID_EDGE_TYPES:
                    continue
                e["from"] = normalize_name(e["from"])
                e["to"] = normalize_name(e["to"])
                corrected = _correct_edge_type(e, name_to_type)
                if corrected is None:
                    continue
                new_edges.append(EdgeCandidate(
                    from_name=corrected["from"],
                    to_name=corrected["to"],
                    type=EdgeType(corrected["type"]),
                    instruction=corrected.get("instruction", ""),
                ))

            return FinalizeResult(
                promoted_patterns=promoted_patterns,
                new_edges=new_edges,
                invalidations=p.get("invalidations", []),
            )
        except (json.JSONDecodeError, Exception):
            return FinalizeResult(promoted_patterns=[], new_edges=[], invalidations=[])
