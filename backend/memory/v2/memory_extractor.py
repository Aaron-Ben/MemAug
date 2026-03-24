"""Memory extraction module."""
import json
import logging
import aiofiles
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml
from jinja2 import Template

from app.services.llm import get_llm
from memory.v2.model import CandidateMemory, MemoryCategory, MergedMemoryPayload, MemoryContext

logger = logging.getLogger(__name__)

DATA_BASE_DIR= Path(__file__).parent.parent.parent / "data"

class RequestContext:
    user: str



class MemoryExtractor:
    """Extract candidate memories from session context."""

    def __init__(self):
        self.llm = get_llm()
        self._prompt_templates: Dict[str, Template] = {}

    def _load_prompt_template(self, name: str = "memory_extraction") -> Template:
        """Load a prompt template by name."""
        if name not in self._prompt_templates:
            prompt_path = Path(__file__).parent / "prompt" / f"{name}.yaml"
            with open(prompt_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            template_str = config["template"]
            self._prompt_templates[name] = Template(template_str)
        return self._prompt_templates[name]

    def _format_message_with_parts(self, m: Dict[str, Any]) -> Optional[str]:
        """Format a message with its parts for context."""
        # Handle simple messages with content field
        if content := m.get("content"):
            return content

        # Handle messages with parts (e.g., tool results)
        if parts := m.get("parts"):
            return "\n".join(str(p) for p in parts)

        return None

    async def extract(
        self,
        context: dict,
        user: str,
        session_id: str,
    ) -> List[CandidateMemory]:
        """Extract candidate memories from session messages.

        Args:
            context: Session context containing "messages" key
            user: User identifier
            session_id: Current session ID

        Returns:
            List of CandidateMemory objects
        """
        messages = context.get("messages", [])

        # Format messages for the prompt
        formatted_lines = []
        for m in messages:
            msg_content = self._format_message_with_parts(m)
            if msg_content:
                role = m.get("role", "user")
                formatted_lines.append(f"[{role}]: {msg_content}")

        formatted_messages = "\n".join(formatted_lines)

        if not formatted_messages:
            logger.warning("No formatted messages, returning empty list")
            return []

        # Render prompt template
        template = self._load_prompt_template("memory_extraction")
        prompt = template.render(
            summary="",
            recent_messages=formatted_messages,
            user=user,
            feedback="",
        )

        try:
            # Call LLM
            from app.utils.json import extract_json

            response = await self.llm.generate_response_async(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            logger.debug(f"LLM response: {response[:500]}...")

            # Parse JSON response
            json_str = extract_json(response)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}, response: {response[:200]}")
                data = {}

            if isinstance(data, list):
                logger.warning(
                    "Memory extraction received list instead of dict; wrapping as memories"
                )
                data = {"memories": data}
            elif not isinstance(data, dict):
                logger.warning(
                    "Memory extraction received unexpected type %s; skipping",
                    type(data).__name__,
                )
                data = {}

            # Convert to CandidateMemory objects
            candidates = []
            for mem in data.get("memories", []):
                category_str = mem.get("category", "patterns")
                try:
                    category = MemoryCategory(category_str)
                except ValueError:
                    category = MemoryCategory.PATTERNS

                candidates.append(
                    CandidateMemory(
                        category=category,
                        abstract=mem.get("abstract", ""),
                        overview=mem.get("overview", ""),
                        content=mem.get("content", ""),
                        source_session=session_id,
                    )
                )

            logger.info(f"Extracted {len(candidates)} candidate memories")
            return candidates

        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")
            return []
        
    async def append_to_profile(
        self,
        candidate: CandidateMemory,
        user: str,
    ) -> Optional[MergedMemoryPayload]:
        profile_dir = DATA_BASE_DIR / "user" / user / "memories"
        profile_path = profile_dir / "profile.md"

        existing_content = ""                                                                     
        if profile_path.exists():                                                                 
            try:                                                                                  
                async with aiofiles.open(profile_path, "r", encoding="utf-8") as f:               
                    existing_content = await f.read()                                             
            except Exception as e:                                                                
                logger.warning(f"Failed to read existing profile: {e}")

        if not existing_content.strip():
            profile_dir.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(profile_path, "w", encoding="utf-8") as f:
                await f.write(candidate.content)
            logger.info(f"Created profile at {profile_path}")
            return MergedMemoryPayload(
                abstract=candidate.abstract,
                overview=candidate.overview,
                content=candidate.content,
                reason="created",
            )
        else:
            # 已有内容，合并
            payload = await self.merge_memory_bundle(
                existing_abstract="",
                existing_overview="",
                existing_content=existing_content,
                new_abstract=candidate.abstract,
                new_overview=candidate.overview,
                new_content=candidate.content,
                category="profile",
            )
            if not payload:
                logger.warning("Profile merge bundle failed; keeping existing profile unchanged")
                return None

            async with aiofiles.open(profile_path, "w", encoding="utf-8") as f:
                await f.write(payload.content)
            logger.info(f"Merged profile info to {profile_path}")
            return payload


    async def merge_memory_bundle(
        self,
        existing_abstract: str,
        existing_overview: str,
        existing_content: str,
        new_abstract: str,
        new_overview: str,
        new_content: str,
        category: str,
    ) -> Optional[MergedMemoryPayload]:
        """Merge existing memory with new memory using one LLM call.

        Args:
            existing_abstract: Existing memory abstract (L0)
            existing_overview: Existing memory overview (L1)
            existing_content: Existing memory content (L2)
            new_abstract: New memory abstract (L0)
            new_overview: New memory overview (L1)
            new_content: New memory content (L2)
            category: Memory category (profile, preferences, etc.)

        Returns:
            Merged memory payload or None if failed
        """
        # Render prompt template
        template = self._load_prompt_template("memory_merge_bundle")
        prompt = template.render(
            existing_abstract=existing_abstract,
            existing_overview=existing_overview,
            existing_content=existing_content,
            new_abstract=new_abstract,
            new_overview=new_overview,
            new_content=new_content,
            category=category,
        )

        try:
            from app.utils.json import extract_json

            response = await self.llm.generate_response_async(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            # Parse JSON response
            json_str = extract_json(response)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}, response: {response[:200]}")
                return None

            if not isinstance(data, dict):
                logger.error("Memory merge bundle parse failed: non-dict payload")
                return None

            abstract = str(data.get("abstract", "") or "").strip()
            overview = str(data.get("overview", "") or "").strip()
            content = str(data.get("content", "") or "").strip()
            reason = str(data.get("reason", "") or "").strip()
            decision = str(data.get("decision", "") or "").strip().lower()

            if decision and decision != "merge":
                logger.error("Memory merge bundle invalid decision=%s", decision)
                return None
            if not abstract or not content:
                logger.error(
                    "Memory merge bundle missing required fields abstract/content: %s",
                    data,
                )
                return None

            return MergedMemoryPayload(
                abstract=abstract,
                overview=overview,
                content=content,
                reason=reason,
            )
        except Exception as e:
            logger.error(f"Memory merge bundle failed: {e}")
            return None

    async def create_memory(
        self,
        candidate: CandidateMemory,
        user: str,
        session_id: str,
    ) -> Optional[MemoryContext]:
        """Create MemoryContext from candidate and persist to file system.

        Args:
            candidate: Candidate memory to create
            user: User identifier
            session_id: Current session ID

        Returns:
            MemoryContext object or None if failed
        """
        # Special handling for profile: append to profile.md
        if candidate.category == MemoryCategory.PROFILE:
            payload = await self.append_to_profile(candidate, user)
            if not payload:
                return None

            memory_uri = f"data/user/{user}/memories/profile.md"
            memory = MemoryContext(
                uri=memory_uri,
                parent_uri=f"data/user/{user}/memories",
                is_leaf=True,
                abstract=payload.abstract,
                category=candidate.category.value,
                session_id=session_id,
                user=user,
            )
            logger.info(f"Created profile memory: {memory_uri}")
            return memory

        # Determine directory based on category
        # preferences/entities/events -> user space
        # cases/patterns -> agent space
        if candidate.category in [
            MemoryCategory.PREFERENCES,
            MemoryCategory.ENTITIES,
            MemoryCategory.EVENTS,
        ]:
            base_dir = DATA_BASE_DIR / "user" / user / "memories" / candidate.category.value
            parent_uri = f"data/user/{user}/memories/{candidate.category.value}"
        else:  # CASES, PATTERNS
            base_dir = DATA_BASE_DIR / "agent" / user / "memories" / candidate.category.value
            parent_uri = f"data/agent/{user}/memories/{candidate.category.value}"

        # Generate file path
        memory_id = f"mem_{uuid4()}"
        memory_path = base_dir / f"{memory_id}.md"
        memory_uri = f"{parent_uri}/{memory_id}.md"

        # Write to file
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(memory_path, "w", encoding="utf-8") as f:
                await f.write(candidate.content)
            logger.info(f"Created memory file: {memory_uri}")
        except Exception as e:
            logger.error(f"Failed to write memory to file: {e}")
            return None

        # Create MemoryContext object
        memory = MemoryContext(
            uri=memory_uri,
            parent_uri=parent_uri,
            is_leaf=True,
            abstract=candidate.abstract,
            category=candidate.category.value,
            session_id=session_id,
            user=user,
        )
        logger.info(f"Created memory: {memory_uri}, abstract: {candidate.abstract}")
        return memory