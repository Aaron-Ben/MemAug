"""Character storage service for file system based character management.

Characters are stored as:
- data/characters/{uuid}/prompt.md - Character system prompt
- data/characters/{uuid}/daily/ - Diary files

Chat history is managed by ChatHistoryService in data/history/.

The character_id IS the UUID - no separate mapping needed.
"""

import uuid
import logging
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from app.models.character import UserCharacter


logger = logging.getLogger(__name__)


# Default characters directory
DEFAULT_CHARACTERS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "characters"


class CharacterStorageService:
    """Service for managing characters using file system storage."""

    def __init__(self, characters_dir: Optional[Path] = None):
        self.characters_dir = characters_dir or DEFAULT_CHARACTERS_DIR
        self.characters_dir.mkdir(parents=True, exist_ok=True)

    def create_character(self, name: str, prompt: str) -> UserCharacter:
        """Create a new character with prompt.md file."""
        if not name or not name.strip():
            raise ValueError("Character name cannot be empty")
        if not prompt or not prompt.strip():
            raise ValueError("Character prompt cannot be empty")

        # Generate UUID as character_id
        character_id = str(uuid.uuid4())

        # Create character directory
        character_dir = self.characters_dir / character_id
        character_dir.mkdir(parents=True, exist_ok=True)

        # Create prompt.md with name as heading
        prompt_content = f"# {name.strip()}\n\n{prompt.strip()}"
        prompt_file = character_dir / "prompt.md"
        prompt_file.write_text(prompt_content, encoding='utf-8')

        # Create subdirectories
        (character_dir / "daily").mkdir(exist_ok=True)

        logger.info(f"Created character: {character_id}")

        return UserCharacter(
            character_id=character_id,
            name=name.strip(),
            created_at=datetime.now().isoformat()
        )

    def list_characters(self) -> List[UserCharacter]:
        """List all characters."""
        characters = []

        for character_dir in self.characters_dir.iterdir():
            if not character_dir.is_dir():
                continue

            try:
                character_id = character_dir.name
                # Validate it's a UUID
                uuid.UUID(character_id)

                prompt_file = character_dir / "prompt.md"
                if not prompt_file.exists():
                    continue

                # Get name from prompt.md first line
                name = character_id
                try:
                    first_line = prompt_file.read_text(encoding='utf-8').split('\n')[0]
                    if first_line.startswith('# '):
                        name = first_line[2:].strip()
                    elif first_line.strip():
                        name = first_line.strip()[:50]
                except:
                    pass

                # Get created_at from directory mtime
                created_at = datetime.fromtimestamp(character_dir.stat().st_ctime).isoformat()

                characters.append(UserCharacter(
                    character_id=character_id,
                    name=name,
                    created_at=created_at
                ))
            except (ValueError, OSError):
                # Not a valid UUID directory, skip
                continue

        characters.sort(key=lambda c: c.created_at, reverse=True)
        return characters

    def get_character(self, character_id: str) -> Optional[UserCharacter]:
        """Get a character by ID."""
        try:
            uuid.UUID(character_id)
        except ValueError:
            return None

        character_dir = self.characters_dir / character_id
        if not character_dir.exists():
            return None

        prompt_file = character_dir / "prompt.md"
        if not prompt_file.exists():
            return None

        # Get name from prompt.md
        name = character_id
        try:
            first_line = prompt_file.read_text(encoding='utf-8').split('\n')[0]
            if first_line.startswith('# '):
                name = first_line[2:].strip()
            elif first_line.strip():
                name = first_line.strip()[:50]
        except:
            pass

        created_at = datetime.fromtimestamp(character_dir.stat().st_ctime).isoformat()

        return UserCharacter(
            character_id=character_id,
            name=name,
            created_at=created_at
        )

    def delete_character(self, character_id: str) -> bool:
        """Delete a character."""
        try:
            uuid.UUID(character_id)
        except ValueError:
            return False

        character_dir = self.characters_dir / character_id
        try:
            if character_dir.exists():
                shutil.rmtree(character_dir)
                logger.info(f"Deleted character: {character_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting character {character_id}: {e}")
            return False

    def update_prompt(self, character_id: str, prompt: str) -> bool:
        """Update a character's prompt."""
        if not prompt or not prompt.strip():
            raise ValueError("Character prompt cannot be empty")

        try:
            uuid.UUID(character_id)
        except ValueError:
            return False

        character_dir = self.characters_dir / character_id
        prompt_file = character_dir / "prompt.md"

        if not character_dir.exists():
            return False

        try:
            # Preserve the name heading, replace the rest
            current_content = prompt_file.read_text(encoding='utf-8')
            lines = current_content.split('\n')
            name_heading = lines[0] if lines and lines[0].startswith('# ') else None

            new_content = prompt.strip()
            if name_heading:
                new_content = f"{name_heading}\n\n{new_content}"

            prompt_file.write_text(new_content, encoding='utf-8')
            logger.info(f"Updated prompt for character: {character_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating prompt for {character_id}: {e}")
            return False

    def get_prompt(self, character_id: str) -> Optional[str]:
        """Get a character's prompt (without the name heading)."""
        try:
            uuid.UUID(character_id)
        except ValueError:
            return None

        character_dir = self.characters_dir / character_id
        prompt_file = character_dir / "prompt.md"

        if not prompt_file.exists():
            return None

        try:
            content = prompt_file.read_text(encoding='utf-8')
            # Remove name heading if present
            lines = content.split('\n')
            if lines and lines[0].startswith('# '):
                return '\n'.join(lines[1:]).lstrip()
            return content
        except Exception as e:
            logger.error(f"Error reading prompt for {character_id}: {e}")
            return None

    def get_character_dir(self, character_id: str) -> Optional[Path]:
        """Get the directory path for a character."""
        try:
            uuid.UUID(character_id)
        except ValueError:
            return None
        return self.characters_dir / character_id

    def get_daily_dir(self, character_id: str) -> Optional[Path]:
        """Get the daily diary directory for a character."""
        character_dir = self.get_character_dir(character_id)
        if not character_dir:
            return None
        return character_dir / "daily"

