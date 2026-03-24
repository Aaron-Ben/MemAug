"""Memory extraction module."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Template

from app.services.llm import get_llm
from memory.v2.model import CandidateMemory, MemoryCategory

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """Extract candidate memories from session context."""

    def __init__(self):
        self.llm = get_llm()
        self._prompt_template: Optional[Template] = None

    def _load_prompt_template(self) -> Template:
        """Load the memory extraction prompt template."""
        if self._prompt_template is None:
            prompt_path = Path(__file__).parent / "prompt" / "memory_extraction.yaml"
            with open(prompt_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            template_str = config["template"]
            self._prompt_template = Template(template_str)
        return self._prompt_template

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
        template = self._load_prompt_template()
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
