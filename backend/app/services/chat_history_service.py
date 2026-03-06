"""Chat history service for file-based topic and message management."""

import os
import json
import time
import logging
from typing import List, Optional, Union, Dict
from pathlib import Path

from app.models.chat import ChatMessage, ChatTopic


logger = logging.getLogger(__name__)


# Default paths
CHARACTERS_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "characters"


class ChatHistoryService:
    """Service for managing chat history using file system storage."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the chat history service."""
        self.characters_dir = data_dir or CHARACTERS_DATA_DIR
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        self.characters_dir.mkdir(parents=True, exist_ok=True)

    def _get_topics_dir(self, character_id: str) -> Path:
        """Get topics directory path."""
        return self.characters_dir / character_id / "topics"

    def _get_history_file(self, character_id: str, topic_id: int) -> Path:
        """Get topic file path for a topic."""
        return self._get_topics_dir(character_id) / f"{topic_id}.json"

    def _ensure_topic_dirs(self, character_id: str):
        """Ensure topics directory exists."""
        topics_dir = self._get_topics_dir(character_id)
        topics_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_message(self, msg_data: Union[Dict, ChatMessage], default_name: str = "Unknown") -> ChatMessage:
        """
        Normalize message data to ChatMessage format.

        Handles backward compatibility with old format:
        - Old: {message_id, role, content, timestamp}
        - New: {id, role, name, content, timestamp}
        """
        if isinstance(msg_data, ChatMessage):
            return msg_data

        # Handle old format with message_id
        if "message_id" in msg_data:
            msg_id = msg_data.get("message_id", f"msg_{msg_data.get('timestamp', int(time.time() * 1000))}_{msg_data.get('role', 'user')}_legacy")
            return ChatMessage(
                id=msg_id,
                role=msg_data.get("role", "user"),
                name=msg_data.get("name", default_name),
                content=msg_data.get("content", ""),
                timestamp=msg_data.get("timestamp", int(time.time() * 1000))
            )

        # Handle new format with id
        return ChatMessage(
            id=msg_data.get("id", ChatMessage.generate_id(msg_data.get("role", "user"))),
            role=msg_data.get("role", "user"),
            name=msg_data.get("name", default_name),
            content=msg_data.get("content", ""),
            timestamp=msg_data.get("timestamp", int(time.time() * 1000))
        )

    def _read_history(self, history_file: Path) -> List[ChatMessage]:
        """Read history from file with backward compatibility."""
        try:
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle both array format and object format with "messages" key
                    if isinstance(data, list):
                        return [self._normalize_message(msg) for msg in data]
                    elif isinstance(data, dict) and "messages" in data:
                        return [self._normalize_message(msg) for msg in data["messages"]]
            return []
        except Exception as e:
            logger.error(f"Error reading history from {history_file}: {e}")
            return []

    def _write_history(self, history_file: Path, messages: List[ChatMessage]):
        """Write history to file with atomic write."""
        try:
            # Ensure parent directory exists
            history_file.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first
            temp_file = history_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                data = [msg.model_dump() for msg in messages]
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Atomic rename
            temp_file.replace(history_file)

            logger.debug(f"Saved {len(messages)} messages to {history_file}")
        except Exception as e:
            logger.error(f"Error writing history to {history_file}: {e}")
            raise

    def create_topic(self, user_id: str, character_id: str) -> int:
        """
        Create a new topic.

        Args:
            user_id: User ID (reserved for future use)
            character_id: Character ID

        Returns:
            topic_id: Unix timestamp (seconds)
        """
        # Use current timestamp as topic_id
        topic_id = int(time.time())

        # Ensure directories exist
        self._ensure_topic_dirs(character_id)

        # Create empty history file
        history_file = self._get_history_file(character_id, topic_id)
        self._write_history(history_file, [])

        logger.info(f"Created topic {topic_id} for user {user_id}, character {character_id}")
        return topic_id

    def delete_topic(self, user_id: str, topic_id: int, character_id: Optional[str] = None) -> bool:
        """
        Delete a topic.

        Args:
            user_id: User ID (reserved for future use)
            topic_id: Topic ID to delete
            character_id: Optional character ID for validation

        Returns:
            bool: True if deleted successfully
        """
        try:
            # If character_id not provided, search for topic
            if character_id is None:
                character_id = self._find_character_for_topic(topic_id)
                if character_id is None:
                    logger.warning(f"Topic {topic_id} not found")
                    return False

            history_file = self._get_history_file(character_id, topic_id)

            if history_file.exists():
                history_file.unlink()
                logger.info(f"Deleted topic {topic_id} for character {character_id}")
                return True
            else:
                logger.warning(f"Topic file not found: {history_file}")
                return False
        except Exception as e:
            logger.error(f"Error deleting topic {topic_id}: {e}")
            return False

    def _find_character_for_topic(self, topic_id: int) -> Optional[str]:
        """Find which character a topic belongs to."""
        if not self.characters_dir.exists():
            return None

        for character_dir in self.characters_dir.iterdir():
            if not character_dir.is_dir():
                continue
            topics_dir = character_dir / "topics"
            if topics_dir.exists() and (topics_dir / f"{topic_id}.json").exists():
                return character_dir.name
        return None

    def list_topics(self, user_id: str, character_id: Optional[str] = None) -> List[ChatTopic]:
        """
        List topics for a user.

        Args:
            user_id: User ID (reserved for future use)
            character_id: Optional character ID to filter by

        Returns:
            List of ChatTopic objects sorted by updated_at (newest first)
        """
        topics = []

        if not self.characters_dir.exists():
            return topics

        for character_dir in self.characters_dir.iterdir():
            if not character_dir.is_dir():
                continue

            # Filter by character_id if provided
            if character_id and character_dir.name != character_id:
                continue

            topics_dir = character_dir / "topics"
            if not topics_dir.exists():
                continue

            for topic_file in topics_dir.iterdir():
                if not topic_file.is_file():
                    continue

                try:
                    topic_id = int(topic_file.stem)

                    # Get timestamps from filesystem
                    stat = topic_file.stat()
                    created_at = int(stat.st_ctime)
                    updated_at = int(stat.st_mtime)

                    # Get message count from file
                    messages = self._read_history(topic_file)
                    message_count = len(messages)

                    topics.append(ChatTopic(
                        topic_id=topic_id,
                        character_id=character_dir.name,
                        created_at=created_at,
                        updated_at=updated_at,
                        message_count=message_count
                    ))
                except (ValueError, OSError) as e:
                    logger.warning(f"Error reading topic {topic_file}: {e}")
                    continue

        # Sort by updated_at (newest first)
        topics.sort(key=lambda t: t.updated_at, reverse=True)
        return topics

    def get_topic_history(self, user_id: str, topic_id: int, character_id: Optional[str] = None) -> List[ChatMessage]:
        """
        Get chat history for a topic.

        Args:
            user_id: User ID (reserved for future use)
            topic_id: Topic ID
            character_id: Optional character ID (required if not in default location)

        Returns:
            List of ChatMessage objects
        """
        # Find character ID if not provided
        if character_id is None:
            character_id = self._find_character_for_topic(topic_id)
            if character_id is None:
                logger.warning(f"Topic {topic_id} not found")
                return []

        history_file = self._get_history_file(character_id, topic_id)
        return self._read_history(history_file)

    def append_message(
        self,
        user_id: str,
        topic_id: int,
        role: str,
        content: str,
        name: str,
        character_id: Optional[str] = None
    ) -> ChatMessage:
        """
        Append a message to a topic.

        Args:
            user_id: User ID (reserved for future use)
            topic_id: Topic ID
            role: Message role ('user' or 'assistant')
            content: Message content
            name: Character/User name
            character_id: Optional character ID

        Returns:
            The created ChatMessage
        """
        # Find character ID if not provided
        if character_id is None:
            character_id = self._find_character_for_topic(topic_id)
            if character_id is None:
                raise ValueError(f"Topic {topic_id} not found")

        # Create new message with new format
        message = ChatMessage(
            id=ChatMessage.generate_id(role),
            role=role,
            name=name,
            content=content,
            timestamp=int(time.time() * 1000)  # Milliseconds
        )

        # Read existing messages
        history_file = self._get_history_file(character_id, topic_id)
        messages = self._read_history(history_file)

        # Append new message
        messages.append(message)

        # Write back to file
        self._write_history(history_file, messages)

        logger.debug(f"Appended message to topic {topic_id}")
        return message

    def get_or_create_default_topic(self, user_id: str, character_id: str) -> int:
        """
        Get existing default topic or create a new one.

        The default topic is the most recently updated topic for the character.

        Args:
            user_id: User ID (reserved for future use)
            character_id: Character ID

        Returns:
            topic_id: Topic ID
        """
        topics = self.list_topics(user_id, character_id)

        if topics:
            # Return the most recently updated topic
            return topics[0].topic_id
        else:
            # Create new topic
            return self.create_topic(user_id, character_id)

    def get_history_for_chat(self, user_id: str, topic_id: Optional[int], character_id: str) -> List[dict]:
        """
        Get chat history formatted for LLM consumption.

        Args:
            user_id: User ID (reserved for future use)
            topic_id: Topic ID (if None, uses default topic)
            character_id: Character ID

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        if topic_id is None:
            topic_id = self.get_or_create_default_topic(user_id, character_id)

        messages = self.get_topic_history(user_id, topic_id, character_id)
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
