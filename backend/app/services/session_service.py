"""Session management service for emotional-companionship.

Manages session lifecycle:
- Message storage and retrieval
- Automatic archiving when threshold reached
- Long-term memory extraction via Compressor
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

from app.models.chat import ChatMessage
from app.services.chat_history_service import ChatHistoryService
from memory.v2.chromadb_manager import ChromaDBManager
from memory.v2.compressor import Compressor
from memory.v2.session_archiver import SessionArchiver

logger = logging.getLogger(__name__)

# Default paths - point to project root data directory
DATA_BASE_DIR = Path(__file__).parent.parent.parent.parent / "data"


@dataclass
class SessionStats:
    """Session statistics information."""

    total_turns: int = 0
    total_tokens: int = 0
    compression_count: int = 0
    memories_extracted: int = 0


class Session:
    """Session - manages a single conversation's lifecycle.

    Acts as a wrapper around ChatHistoryService, adding:
    - Automatic archiving (commit when threshold reached)
    - Memory extraction (via Compressor)
    - Usage tracking (contexts and skills)
    - Summary/overview generation
    """

    def __init__(
        self,
        character_id: str,
        topic_id: int,
        chat_history: ChatHistoryService,
        compressor: Optional[Compressor] = None,
        auto_commit_threshold: int = 6,
        user_id: str = "user_default",
        archiver: Optional[SessionArchiver] = None,
    ):
        self._character_id = character_id
        self._topic_id = topic_id
        self._chat_history = chat_history
        self._compressor = compressor
        self._auto_commit_threshold = auto_commit_threshold
        self._user_id = user_id
        self._archiver = archiver or SessionArchiver()

        self._messages: List[ChatMessage] = []
        self._stats = SessionStats()
        self._compression_index = 0
        self._loaded = False

        # Session file path: data/session/{user_id}/{session_id}
        self._session_dir = DATA_BASE_DIR / "session" / self._user_id / self.session_id
        # History is at data/session/{user_id}/{session_id}/history/
        self._history_dir = self._session_dir / "history"

    @property
    def session_id(self) -> str:
        """Get session ID (topic_id as string)."""
        return str(self._topic_id)

    @property
    def character_id(self) -> str:
        """Get character ID."""
        return self._character_id

    @property
    def topic_id(self) -> int:
        """Get topic ID."""
        return self._topic_id

    @property
    def messages(self) -> List[ChatMessage]:
        """Get current messages."""
        return self._messages

    @property
    def stats(self) -> SessionStats:
        """Get session statistics."""
        return self._stats

    @property
    def compression_index(self) -> int:
        """Get compression/archive index."""
        return self._compression_index

    async def load(self) -> None:
        """Load session data from storage."""
        if self._loaded:
            return

        # Load messages from chat history
        messages = self._chat_history.get_topic_history(
            self._user_id, self._topic_id, self._character_id
        )
        self._messages = messages

        # Restore compression_index from history
        await self._restore_compression_index()

        self._loaded = True
        logger.info(f"Session loaded: {self.session_id} ({len(self._messages)} messages)")

    async def _restore_compression_index(self) -> None:
        """Scan history directory to restore compression_index."""
        if not self._history_dir.exists():
            return

        try:
            archives = [d.name for d in self._history_dir.iterdir() if d.is_dir() and d.name.startswith("archive_")]
            if archives:
                max_index = max(int(a.split("_")[1]) for a in archives)
                self._compression_index = max_index
                self._stats.compression_count = len(archives)
                logger.debug(f"Restored compression_index: {max_index}")
        except Exception as e:
            logger.warning(f"Failed to restore compression_index: {e}")

    async def exists(self) -> bool:
        """Check if session exists in storage."""
        return self._session_dir.exists()

    async def ensure_exists(self) -> None:
        """Ensure session directory and files exist."""
        if await self.exists():
            return

        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Create empty history directory
        self._history_dir.mkdir(parents=True, exist_ok=True)

        # Initialize messages file via chat history
        self._chat_history.create_topic(self._user_id, self._character_id)

    def add_message(
        self,
        role: str,
        content: str,
        name: str,
    ) -> ChatMessage:
        """Add a message to the session.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            name: Character/User name

        Returns:
            The created ChatMessage
        """
        msg = self._chat_history.append_message(
            user_id=self._user_id,
            topic_id=self._topic_id,
            role=role,
            content=content,
            name=name,
            character_id=self._character_id,
        )

        self._messages.append(msg)

        # Update statistics
        if role == "user":
            self._stats.total_turns += 1
        self._stats.total_tokens += len(content) // 4

        logger.debug(f"Added message to session {self.session_id}: {role}")
        return msg

    async def commit(self) -> Dict[str, Any]:
        """Commit session: archive messages and extract long-term memories.

        Two-phase approach:
        1. Archive: Write to history/, clear current messages
        2. Extract: Use Compressor to extract memories

        Returns:
            Dict with commit results
        """
        result = {
            "session_id": self.session_id,
            "status": "committed",
            "memories_extracted": 0,
            "archived": False,
            "stats": None,
        }

        if not self._messages:
            logger.debug("No messages to commit")
            return result

        # ===== Phase 1: Archive =====
        self._compression_index += 1
        messages_to_archive = self._messages.copy()

        summary = await self._archiver.generate_archive_summary(messages_to_archive)
        abstract = self._extract_abstract_from_summary(summary)
        overview = summary

        # Write archive
        archive_dir = self._history_dir / f"archive_{self._compression_index:03d}"
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Write messages.jsonl
        lines = [self._message_to_jsonl(m) for m in messages_to_archive]
        async with aiofiles.open(archive_dir / "messages.jsonl", "w", encoding="utf-8") as f:
            await f.write("\n".join(lines) + "\n")

        # Write .abstract.md and .overview.md
        async with aiofiles.open(archive_dir / ".abstract.md", "w", encoding="utf-8") as f:
            await f.write(abstract)
        async with aiofiles.open(archive_dir / ".overview.md", "w", encoding="utf-8") as f:
            await f.write(overview)

        # Clear current messages
        self._messages.clear()
        self._stats.compression_count = self._compression_index
        self._stats.total_tokens = 0

        # Write current session files
        await self._write_session_files()

        result["archived"] = True
        logger.info(
            f"Archived: {len(messages_to_archive)} messages -> "
            f"history/archive_{self._compression_index:03d}/"
        )

        # ===== Phase 2: Memory extraction =====
        if self._compressor:
            logger.info(f"Starting memory extraction from {len(messages_to_archive)} archived messages")

            # Convert messages to dict format for compressor
            messages_dict = [self._message_to_dict(m) for m in messages_to_archive]

            memories = await self._compressor.extract_long_term_memories(
                messages=messages_dict,
                user=self._user_id,
                session_id=self.session_id,
            )

            logger.info(f"Extracted {len(memories)} memories")
            result["memories_extracted"] = len(memories)
            self._stats.memories_extracted += len(memories)

        # Update result stats
        result["stats"] = {
            "total_turns": self._stats.total_turns,
            "memories_extracted": self._stats.memories_extracted,
        }

        logger.info(f"Session {self.session_id} committed")
        return result

    def _message_to_jsonl(self, msg: ChatMessage) -> str:
        """Convert ChatMessage to JSONL format."""
        return json.dumps(msg.model_dump(), ensure_ascii=False)

    def _message_to_dict(self, msg: ChatMessage) -> Dict[str, Any]:
        """Convert ChatMessage to dict format for compressor."""
        return {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
        }

    def _extract_abstract_from_summary(self, summary: str) -> str:
        """Extract one-sentence overview from summary."""
        if not summary:
            return ""

        # Try to extract the overview line
        match = re.search(r"^\*\*([^*]+)\*\*:\s*(.+)$", summary, re.MULTILINE)
        if match:
            return match.group(2).strip()

        first_line = summary.split("\n")[0].strip()
        return first_line if first_line else ""

    async def _write_session_files(self) -> None:
        """Write session-level files (abstract, overview)."""
        turn_count = self._stats.total_turns
        abstract = self._generate_abstract()
        overview = self._generate_overview(turn_count)

        async with aiofiles.open(self._session_dir / ".abstract.md", "w", encoding="utf-8") as f:
            await f.write(abstract)
        async with aiofiles.open(self._session_dir / ".overview.md", "w", encoding="utf-8") as f:
            await f.write(overview)

    def _generate_abstract(self) -> str:
        """Generate one-sentence summary for current session."""
        if not self._messages:
            return ""

        first = self._messages[0].content
        turn_count = self._stats.total_turns
        return f"{turn_count} turns, starting from '{first[:50]}...'"

    def _generate_overview(self, turn_count: int) -> str:
        """Generate session directory structure description."""
        parts = [
            "# Session Directory Structure",
            "",
            "## File Description",
            f"- `messages` - Current messages ({turn_count} turns)",
        ]
        if self._compression_index > 0:
            parts.append(f"- `history/` - Historical archives ({self._compression_index} total)")
        parts.extend([
            "",
            "## Access Methods",
            f"- Full conversation: session/{self._user_id}/{self.session_id}",
        ])
        if self._compression_index > 0:
            parts.append(f"- Historical archives: session/{self._user_id}/{self.session_id}/history/")
        return "\n".join(parts)

    async def get_context_for_search(
        self, query: str, max_archives: int = 3, max_messages: int = 20
    ) -> Dict[str, Any]:
        """Get session context for search/intent analysis.

        Args:
            query: Query string for matching relevant archives
            max_archives: Maximum number of archives to retrieve
            max_messages: Maximum number of recent messages to retrieve

        Returns:
            Dict with 'summaries' and 'recent_messages' keys
        """
        # Recent messages
        recent_messages = list(self._messages[-max_messages:]) if self._messages else []

        # Find relevant archives
        summaries = []
        if self._compression_index > 0 and self._history_dir.exists():
            try:
                query_lower = query.lower()
                scored_archives = []

                for archive_dir in self._history_dir.iterdir():
                    if not archive_dir.is_dir() or not archive_dir.name.startswith("archive_"):
                        continue

                    overview_file = archive_dir / ".overview.md"
                    if not overview_file.exists():
                        continue

                    async with aiofiles.open(overview_file, "r", encoding="utf-8") as f:
                        overview = await f.read()

                    # Calculate relevance by keyword matching
                    score = 0
                    if query_lower in overview.lower():
                        score = overview.lower().count(query_lower)

                    # Time from archive name (higher = newer)
                    try:
                        archive_num = int(archive_dir.name.split("_")[1])
                    except ValueError:
                        archive_num = 0

                    scored_archives.append((score, archive_num, overview))

                # Sort by relevance, then time
                scored_archives.sort(key=lambda x: (x[0], x[1]), reverse=True)
                summaries = [overview for _, _, overview in scored_archives[:max_archives]]

            except Exception as e:
                logger.warning(f"Failed to get archive summaries: {e}")

        return {
            "summaries": summaries,
            "recent_messages": recent_messages,
        }

    def __repr__(self) -> str:
        return f"Session(character={self._character_id}, topic={self._topic_id})"


class SessionService:
    """Session management service.

    Provides high-level API for session operations:
    - Create/load sessions
    - Add messages with auto-commit
    - Manual commit
    - Context retrieval
    """

    def __init__(
        self,
        auto_commit_threshold: int = 6,
        chromadb_manager: Optional[ChromaDBManager] = None,
    ):
        """Initialize session service.

        Args:
            auto_commit_threshold: Message count threshold for auto-commit
            chromadb_manager: Optional ChromaDB manager for memory extraction
        """
        self._chat_history = ChatHistoryService()
        self._auto_commit_threshold = auto_commit_threshold

        # Initialize compressor if ChromaDB provided
        self._compressor = None
        if chromadb_manager:
            self._compressor = Compressor(chromadb=chromadb_manager)

        # Initialize archiver for structured summary generation
        self._archiver = SessionArchiver()

        # Cache for loaded sessions
        self._sessions: Dict[str, Session] = {}

    def _get_session_key(self, character_id: str, topic_id: int) -> str:
        """Get session cache key."""
        return f"{character_id}:{topic_id}"

    async def create_session(self, character_id: str, user_id: str = "user_default") -> Session:
        """Create a new session (topic).

        Args:
            character_id: Character ID
            user_id: User ID

        Returns:
            Newly created Session
        """
        topic_id = self._chat_history.create_topic(user_id, character_id)

        session = Session(
            character_id=character_id,
            topic_id=topic_id,
            chat_history=self._chat_history,
            compressor=self._compressor,
            auto_commit_threshold=self._auto_commit_threshold,
            user_id=user_id,
            archiver=self._archiver,
        )

        await session.ensure_exists()

        key = self._get_session_key(character_id, topic_id)
        self._sessions[key] = session

        logger.info(f"Created session: {character_id}/{topic_id}")
        return session

    async def load_session(
        self,
        character_id: str,
        topic_id: int,
        user_id: str = "user_default",
    ) -> Session:
        """Load an existing session.

        Args:
            character_id: Character ID
            topic_id: Topic ID
            user_id: User ID

        Returns:
            Loaded Session
        """
        key = self._get_session_key(character_id, topic_id)

        # Return cached session if available
        if key in self._sessions:
            session = self._sessions[key]
            if not session._loaded:
                await session.load()
            return session

        # Create new session and load
        session = Session(
            character_id=character_id,
            topic_id=topic_id,
            chat_history=self._chat_history,
            compressor=self._compressor,
            auto_commit_threshold=self._auto_commit_threshold,
            user_id=user_id,
            archiver=self._archiver,
        )

        await session.load()

        self._sessions[key] = session
        logger.info(f"Loaded session: {character_id}/{topic_id}")
        return session

    async def add_message(
        self,
        character_id: str,
        topic_id: int,
        role: str,
        content: str,
        name: str,
        user_id: str = "user_default",
    ) -> ChatMessage:
        """Add a message to session, with auto-commit check.

        Args:
            character_id: Character ID
            topic_id: Topic ID (use None for default topic)
            role: Message role
            content: Message content
            name: Character/User name
            user_id: User ID

        Returns:
            Created ChatMessage
        """
        # Load or get session
        if topic_id is None:
            topic_id = self._chat_history.get_or_create_default_topic(user_id, character_id)

        session = await self.load_session(character_id, topic_id, user_id)

        # Add message
        msg = session.add_message(role, content, name)

        # Check auto-commit threshold
        message_count = len(session.messages)
        if message_count >= self._auto_commit_threshold:
            logger.info(f"Auto-commit triggered: {message_count} >= {self._auto_commit_threshold}")
            await session.commit()

        return msg

    async def commit(
        self,
        character_id: str,
        topic_id: int,
        user_id: str = "user_default",
    ) -> Dict[str, Any]:
        """Manually commit a session.

        Args:
            character_id: Character ID
            topic_id: Topic ID
            user_id: User ID

        Returns:
            Commit result dict
        """
        session = await self.load_session(character_id, topic_id, user_id)
        return await session.commit()

    async def get_context_for_search(
        self,
        character_id: str,
        topic_id: int,
        query: str,
        user_id: str = "user_default",
    ) -> Dict[str, Any]:
        """Get session context for search.

        Args:
            character_id: Character ID
            topic_id: Topic ID
            query: Search query
            user_id: User ID

        Returns:
            Context dict with 'summaries' and 'recent_messages'
        """
        session = await self.load_session(character_id, topic_id, user_id)
        return await session.get_context_for_search(query)

    async def list_sessions(
        self,
        character_id: Optional[str] = None,
        user_id: str = "user_default",
    ) -> List[Dict[str, Any]]:
        """List all sessions (topics).

        Args:
            character_id: Optional character ID filter
            user_id: User ID

        Returns:
            List of session info dicts
        """
        topics = self._chat_history.list_topics(user_id, character_id)
        return [
            {
                "topic_id": t.topic_id,
                "character_id": t.character_id,
                "created_at": t.created_at,
                "updated_at": t.updated_at,
                "message_count": t.message_count,
            }
            for t in topics
        ]