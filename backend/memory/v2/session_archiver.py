"""Session archiver module for generating structured summaries."""

import logging
from pathlib import Path
from typing import Dict, List

import yaml
from jinja2 import Template

from app.models.chat import ChatMessage
from app.services.llm import get_llm

logger = logging.getLogger(__name__)


class SessionArchiver:
    """Generate structured summaries for archived sessions using LLM."""

    def __init__(self):
        self.llm = get_llm()
        self._prompt_templates: Dict[str, Template] = {}

    def _load_prompt_template(self, name: str = "structured_summary") -> Template:
        """Load a prompt template by name."""
        if name not in self._prompt_templates:
            prompt_path = Path(__file__).parent / "prompt" / f"{name}.yaml"
            with open(prompt_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            template_str = config["template"]
            self._prompt_templates[name] = Template(template_str)
        return self._prompt_templates[name]

    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """Format messages for the prompt."""
        lines = []
        for m in messages:
            # Truncate long messages
            content = m.content[:500] + "..." if len(m.content) > 500 else m.content
            lines.append(f"[{m.role}]: {content}")
        return "\n".join(lines)

    async def generate_archive_summary(
        self, messages: List[ChatMessage]
    ) -> str:
        """Generate structured summary for archived messages using LLM.

        Args:
            messages: List of ChatMessage to summarize

        Returns:
            Structured summary string
        """
        if not messages:
            return ""

        # Format messages for the prompt
        formatted_messages = self._format_messages(messages)

        # Load and render prompt template
        template = self._load_prompt_template("structured_summary")
        prompt = template.render(messages=formatted_messages)

        try:
            # Call LLM - structured_summary returns Markdown, not JSON
            response = await self.llm.generate_response_async(
                messages=[{"role": "user", "content": prompt}],
            )

            logger.debug(f"Archive summary generated: {len(response)} chars")
            return response

        except Exception as e:
            logger.error(f"Failed to generate archive summary: {e}")
            # Fallback to simple format
            return self._fallback_summary(messages)

    def _fallback_summary(self, messages: List[ChatMessage]) -> str:
        """Generate simple fallback summary if LLM fails."""
        turn_count = len([m for m in messages if m.role == "user"])
        formatted = "\n".join(
            f"[{m.role}]: {m.content[:200]}..." if len(m.content) > 200 else f"[{m.role}]: {m.content}"
            for m in messages
        )
        return f"# Session Summary\n\n**Overview**: {turn_count} turns, {len(messages)} messages\n\n**Messages**:\n{formatted}"
