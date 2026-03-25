"""Memory deduplication module."""
import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Template

from app.services.llm import get_llm
from memory.v2.chromadb_manager import ChromaDBManager
from memory.v2.model import (
    CandidateMemory,
    DedupDecision,
    DedupResult,
    ExistingMemoryAction,
    MemoryActionDecision,
    MemoryContext,
)

logger = logging.getLogger(__name__)


class MemoryDeduplicator:
    """Handle memory deduplication decisions using LLM."""

    MAX_PROMPT_SIMILAR_MEMORIES = 5
    SIMILARITY_THRESHOLD = 0.7

    # Category constants
    USER_CATEGORIES = {"profile", "preferences", "entities", "events"}
    AGENT_CATEGORIES = {"cases", "patterns"}

    def __init__(self, chromadb_manager: ChromaDBManager):
        """Initialize deduplicator.

        Args:
            chromadb_manager: ChromaDB manager for vector search
        """
        self.chroma_db = chromadb_manager
        self.llm = get_llm()
        self._prompt_templates: Dict[str, Template] = {}
        # Lazy import embedder to avoid circular dependency
        self._embedder = None

    @property
    def embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from app.services.embedding import get_embeddings_batch
            self._embedder = get_embeddings_batch
        return self._embedder

    def _category_uri_prefix(self, category: str, user: str) -> str:
        """Build category URI prefix for file path."""
        if category in self.USER_CATEGORIES:
            return f"data/user/{user}/memories/{category}/"
        elif category in self.AGENT_CATEGORIES:
            return f"data/agent/{user}/memories/{category}/"
        return ""

    def _load_prompt_template(self, name: str = "dedup_decision") -> Template:
        """Load a prompt template by name."""
        if name not in self._prompt_templates:
            prompt_path = Path(__file__).parent / "prompt" / f"{name}.yaml"
            with open(prompt_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            template_str = config["template"]
            self._prompt_templates[name] = Template(template_str)
        return self._prompt_templates[name]

    async def llm_decision(
        self,
        candidate: CandidateMemory,
        similar_memories: List[MemoryContext],
    ) -> tuple[DedupDecision, str, List[ExistingMemoryAction]]:
        """Make deduplication decision using LLM.

        Args:
            candidate: The candidate memory to decide on
            similar_memories: List of similar existing memories

        Returns:
            Tuple of (decision, reason, list of actions)
        """
        # If no similar memories, default to CREATE
        if not similar_memories:
            return DedupDecision.CREATE, "No similar memories found", []

        # Format existing memories for prompt
        existing_formatted = []
        for i, mem in enumerate(similar_memories[: self.MAX_PROMPT_SIMILAR_MEMORIES]):
            abstract = (
                getattr(mem, "abstract", "")
                or getattr(mem, "_abstract_cache", "")
                or (mem.meta or {}).get("abstract", "")
            )
            facet = self._extract_facet_key(abstract)
            score = (mem.meta or {}).get("_dedup_score")
            score_text = "n/a" if score is None else f"{float(score):.4f}"
            existing_formatted.append(
                f"{i + 1}. uri={mem.uri}\n   score={score_text}\n   facet={facet}\n   abstract={abstract}"
            )

        # Render prompt template
        template = self._load_prompt_template("dedup_decision")
        prompt = template.render(
            candidate_content=candidate.content,
            candidate_abstract=candidate.abstract,
            candidate_overview=candidate.overview,
            existing_memories="\n".join(existing_formatted),
        )

        try:
            from app.utils.json import extract_json

            response = await self.llm.generate_response_async(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            logger.debug("Dedup LLM raw response: %s", response[:500])

            # Parse JSON response
            json_str = extract_json(response)
            import json

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}, response: {response[:200]}")
                data = {}

            logger.debug("Dedup LLM parsed payload: %s", data)
            return self._parse_decision_payload(data, similar_memories, candidate)

        except Exception as e:
            logger.warning(f"LLM dedup decision failed: {e}")
            return DedupDecision.CREATE, f"LLM failed: {e}", []

    def _parse_decision_payload(
        self,
        data: dict,
        similar_memories: List[MemoryContext],
        candidate: Optional[CandidateMemory] = None,
    ) -> tuple[DedupDecision, str, List[ExistingMemoryAction]]:
        """Parse/normalize dedup payload from LLM."""
        decision_str = str(data.get("decision", "create")).lower().strip()
        reason = str(data.get("reason", "") or "")

        # Map decision strings to DedupDecision enum
        decision_map = {
            "skip": DedupDecision.SKIP,
            "create": DedupDecision.CREATE,
            "none": DedupDecision.NONE,
            # Legacy: merge maps to none (candidate not stored, but existing updated)
            "merge": DedupDecision.NONE,
        }
        decision = decision_map.get(decision_str, DedupDecision.CREATE)

        # Parse per-memory actions
        raw_actions = data.get("list", [])
        if not isinstance(raw_actions, list):
            raw_actions = []

        # Legacy response compatibility
        if decision_str == "merge" and not raw_actions and similar_memories:
            raw_actions = [
                {
                    "uri": similar_memories[0].uri,
                    "decide": "merge",
                    "reason": "Legacy candidate merge mapped to none",
                }
            ]
            if not reason:
                reason = "Legacy candidate merge mapped to none"

        # Build actions map
        action_map = {
            "merge": MemoryActionDecision.MERGE,
            "delete": MemoryActionDecision.DELETE,
        }
        similar_by_uri: Dict[str, MemoryContext] = {m.uri: m for m in similar_memories}
        actions: List[ExistingMemoryAction] = []
        seen: Dict[str, MemoryActionDecision] = {}

        for item in raw_actions:
            if not isinstance(item, dict):
                continue

            action_str = str(item.get("decide", "")).lower().strip()
            action = action_map.get(action_str)
            if not action:
                continue

            # Find memory by URI
            memory = None
            uri = item.get("uri")
            if isinstance(uri, str):
                memory = similar_by_uri.get(uri)

            # Also support index-based responses
            if memory is None:
                index = item.get("index")
                if isinstance(index, int):
                    if 1 <= index <= len(similar_memories):
                        memory = similar_memories[index - 1]
                    elif 0 <= index < len(similar_memories):
                        memory = similar_memories[index]

            if memory is None:
                continue

            # Handle conflicting actions
            previous = seen.get(memory.uri)
            if previous and previous != action:
                actions = [a for a in actions if a.memory.uri != memory.uri]
                seen.pop(memory.uri, None)
                logger.warning(f"Conflicting actions for memory {memory.uri}, dropping both")
                continue
            if previous == action:
                continue

            seen[memory.uri] = action
            actions.append(
                ExistingMemoryAction(
                    memory=memory,
                    decision=action,
                    reason=str(item.get("reason", "") or ""),
                )
            )

        # Rule: skip should never carry per-memory actions
        if decision == DedupDecision.SKIP:
            return decision, reason, []

        # Check if any merge action exists
        has_merge_action = any(a.decision == MemoryActionDecision.MERGE for a in actions)

        # Rule: if any merge exists, ignore create and execute as none
        if decision == DedupDecision.CREATE and has_merge_action:
            decision = DedupDecision.NONE
            reason = f"{reason} | normalized:create+merge->none".strip(" |")
            return decision, reason, actions

        # Rule: create can only carry delete actions (or empty list)
        if decision == DedupDecision.CREATE:
            actions = [a for a in actions if a.decision == MemoryActionDecision.DELETE]

        return decision, reason, actions

    @staticmethod
    def _extract_facet_key(text: str) -> str:
        """Extract normalized facet key from memory abstract (before separator)."""
        if not text:
            return ""

        normalized = " ".join(str(text).strip().split())
        # Prefer common separators used by extraction templates
        for sep in ("：", ":", "-", "—"):
            if sep in normalized:
                left = normalized.split(sep, 1)[0].strip().lower()
                if left:
                    return left

        # Fallback: short leading phrase
        m = re.match(r"^(.{1,24})\s", normalized.lower())
        if m:
            return m.group(1).strip()
        return normalized[:24].lower().strip()

    async def deduplicate(
        self,
        candidate: CandidateMemory,
        *,
        batch_memories: Optional[List[tuple[List[float], MemoryContext]]] = None,
    ) -> DedupResult:
        """Decide how to handle a candidate memory.

        Args:
            candidate: The candidate memory to deduplicate
            batch_memories: Optional batch-internal memories with their vectors

        Returns:
            DedupResult with decision and actions
        """
        # Step 1: Vector pre-filtering - find similar memories in same category
        similar_memories, query_vector = await self.find_similar_memories(
            candidate, batch_memories=batch_memories
        )

        if not similar_memories:
            # No similar memories, create directly
            return DedupResult(
                decision=DedupDecision.CREATE,
                candidate=candidate,
                similar_memories=[],
                actions=[],
                reason="No similar memories found",
                query_vector=query_vector,
            )

        # Step 2: LLM decision
        decision, reason, actions = await self.llm_decision(candidate, similar_memories)

        return DedupResult(
            decision=decision,
            candidate=candidate,
            similar_memories=similar_memories,
            actions=None if decision == DedupDecision.SKIP else actions,
            reason=reason,
            query_vector=query_vector,
        )

    async def find_similar_memories(
        self,
        candidate: CandidateMemory,
        *,
        batch_memories: Optional[List[tuple[List[float], MemoryContext]]] = None,
    ) -> tuple[List[MemoryContext], List[float]]:
        """Find similar existing memories using vector search.

        Returns:
            Tuple of (similar_memories, query_vector)
        """
        query_vector: List[float] = []

        # Generate embedding for candidate
        query_text = f"{candidate.abstract} {candidate.content}"
        query_vector = self.embedder(query_text)

        # Build category URI prefix
        category_uri_prefix = self._category_uri_prefix(
            candidate.category.value, candidate.user
        )

        owner = candidate.user
        owner_space = owner

        try:
            results = await self.chroma_db.search_similar_memories(
                owner_space=owner_space,
                category_uri_prefix=category_uri_prefix,
                query_vector=query_vector,
                limit=5,
            )

            similar = []
            for result in results:
                score = float(result.get("_score", 0))
                if score >= self.SIMILARITY_THRESHOLD:
                    context = MemoryContext.from_dict(result)
                    if context:
                        context.meta = {**(context.meta or {}), "_dedup_score": score}
                        similar.append(context)

            if batch_memories:
                seen_uris = {c.uri for c in similar}
                for batch_vec, batch_ctx in batch_memories:
                    if batch_ctx.uri in seen_uris:
                        continue
                    score = self._cosine_similarity(query_vector, batch_vec)
                    if score >= self.SIMILARITY_THRESHOLD:
                        ctx_copy = copy.copy(batch_ctx)
                        ctx_copy.meta = {**(batch_ctx.meta or {}), "_dedup_score": score}
                        similar.append(ctx_copy)

            return similar, query_vector
            
        except Exception as e:
            logger.warning(f"Vector search failed:{e}")
            return [], query_vector

    def result_to_memory_context(self, result: Dict[str, Any]) -> Optional[MemoryContext]:
        """Convert search result dict to MemoryContext."""
        try:
            return MemoryContext(
                id=result.get("id", ""),
                uri=result.get("uri", ""),
                parent_uri=result.get("parent_uri", ""),
                category=result.get("category", ""),
                abstract=result.get("abstract", ""),
                overview=result.get("overview", ""),
                content=result.get("content", ""),
                level=result.get("level", 2),
                vector=result.get("vector"),
                session_id=result.get("session_id", ""),
                user=result.get("user", ""),
                is_leaf=result.get("is_leaf", True),
            )
        except Exception as e:
            logger.warning(f"Failed to convert result to MemoryContext: {e}")
            return None

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec_a) != len(vec_b):
            return 0.0

        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = sum(a * a for a in vec_a) ** 0.5
        mag_b = sum(b * b for b in vec_b) ** 0.5

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)
