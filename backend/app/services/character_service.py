"""Character storage service for file system based character management.

Characters are stored as:
- data/characters/{uuid}/prompt.md - Character system prompt
- data/characters/{uuid}/topics/ - Conversation topics
- data/characters/{uuid}/.character_meta.json - Metadata (name, created_at, updated_at)
- data/daily/{name}/ - Diary files (global directory organized by name)
"""

import json
import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

from app.models.character import Character


logger = logging.getLogger(__name__)


# Default directories（指向项目根目录的 data 文件夹）
DEFAULT_CHARACTERS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "characters"
DEFAULT_DAILY_DIR = Path(__file__).parent.parent.parent.parent / "data" / "daily"


class CharacterService:
    """Service for managing characters using file system storage."""

    def __init__(self, characters_dir: Optional[Path] = None, daily_dir: Optional[Path] = None):
        self.characters_dir = characters_dir or DEFAULT_CHARACTERS_DIR
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        self.daily_dir = daily_dir or DEFAULT_DAILY_DIR
        self.daily_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_name(self, name: str) -> str:
        """Sanitize character name for use as directory name."""
        # Remove file system invalid characters
        sanitized = re.sub(r'[\\/:*?"<>|]', '', name.strip())
        # Replace whitespace with underscores
        sanitized = re.sub(r'\s+', '_', sanitized)
        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized or 'unnamed'

    def create_character(self, name: str, prompt: str) -> Character:
        """Create a new character with UUID directory and metadata file."""
        if not name or not name.strip():
            raise ValueError("Character name cannot be empty")
        if not prompt or not prompt.strip():
            raise ValueError("Character prompt cannot be empty")

        # Generate UUID4 as character_id
        character_id = str(uuid.uuid4())

        # Create character directory
        character_dir = self.characters_dir / character_id
        character_dir.mkdir(parents=True, exist_ok=True)

        # Create prompt.md with character info
        prompt_content = f"""## 角色信息
- character_id: {character_id}
- 角色名称: {name.strip()}

{prompt.strip()}"""
        prompt_file = character_dir / "prompt.md"
        prompt_file.write_text(prompt_content, encoding='utf-8')

        # Create subdirectories
        (character_dir / "topics").mkdir(exist_ok=True)

        # Create metadata file
        now = datetime.now().isoformat()
        metadata = {
            "character_id": character_id,
            "name": name.strip(),
            "created_at": now,
            "updated_at": now
        }
        meta_file = character_dir / ".character_meta.json"
        meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

        # Create daily directory (global directory by name)
        daily_name_dir = self.daily_dir / self._sanitize_name(name.strip())
        daily_name_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created character: {character_id} ({name.strip()})")

        return Character(
            character_id=character_id,
            name=name.strip(),
            created_at=now,
            updated_at=now
        )

    def _load_metadata(self, character_dir: Path) -> Optional[Dict[str, any]]:
        """Load character metadata from .character_meta.json file."""
        meta_file = character_dir / ".character_meta.json"
        if meta_file.exists():
            try:
                return json.loads(meta_file.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def _save_metadata(self, character_dir: Path, metadata: Dict[str, any]) -> None:
        """Save character metadata to .character_meta.json file."""
        meta_file = character_dir / ".character_meta.json"
        meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


    def list_characters(self) -> List[Character]:
        """List all characters."""
        characters = []

        for character_dir in self.characters_dir.iterdir():
            if not character_dir.is_dir():
                continue

            try:
                character_id = character_dir.name

                prompt_file = character_dir / "prompt.md"
                if not prompt_file.exists():
                    continue

                # Load from metadata file
                metadata = self._load_metadata(character_dir)
                if metadata:
                    characters.append(Character(
                        character_id=character_id,
                        name=metadata.get("name", character_id),
                        created_at=metadata.get("created_at", datetime.fromtimestamp(character_dir.stat().st_ctime).isoformat()),
                        updated_at=metadata.get("updated_at", metadata.get("created_at", ""))
                    ))
            except OSError:
                continue

        characters.sort(key=lambda c: c.created_at, reverse=True)
        return characters

    def get_character(self, character_id: str) -> Optional[Character]:
        """Get a character by ID (UUID)."""
        character_dir = self.characters_dir / character_id
        if not character_dir.exists():
            return None

        prompt_file = character_dir / "prompt.md"
        if not prompt_file.exists():
            return None

        # Load from metadata file
        metadata = self._load_metadata(character_dir)
        if metadata:
            return Character(
                character_id=character_id,
                name=metadata.get("name", character_id),
                created_at=metadata.get("created_at", datetime.fromtimestamp(character_dir.stat().st_ctime).isoformat()),
                updated_at=metadata.get("updated_at", metadata.get("created_at", ""))
            )

        return None

    def get_character_by_name(self, name: str) -> Optional[Character]:
        """Get a character by display name."""
        sanitized = self._sanitize_name(name)
        for character in self.list_characters():
            if self._sanitize_name(character.name) == sanitized:
                return character
        return None

    def delete_character(self, character_id: str) -> bool:
        """Delete a character."""
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
        """Update a character's prompt and update timestamp."""
        if not prompt or not prompt.strip():
            raise ValueError("Character prompt cannot be empty")

        character_dir = self.characters_dir / character_id
        prompt_file = character_dir / "prompt.md"

        if not character_dir.exists():
            return False

        try:
            # Get character info from metadata
            metadata = self._load_metadata(character_dir)

            # Preserve the character info block, replace the prompt content
            current_content = prompt_file.read_text(encoding='utf-8')
            lines = current_content.split('\n')

            # Find the end of character info block (after the role name line)
            info_block_end = -1
            for i, line in enumerate(lines):
                if i >= 3 and line.strip() and not line.startswith('-'):
                    info_block_end = i
                    break

            if info_block_end == -1:
                # If pattern not found, prepend character info
                character_id = metadata.get("character_id", character_id) if metadata else character_id
                name = metadata.get("name", "") if metadata else ""
                new_content = f"""## 角色信息
- character_id: {character_id}
- 角色名称: {name}

{prompt.strip()}"""
            else:
                # Replace content after character info block
                new_content = '\n'.join(lines[:info_block_end]) + '\n\n' + prompt.strip()

            prompt_file.write_text(new_content, encoding='utf-8')

            # Update metadata timestamp
            if metadata:
                metadata["updated_at"] = datetime.now().isoformat()
                self._save_metadata(character_dir, metadata)

            logger.info(f"Updated prompt for character: {character_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating prompt for {character_id}: {e}")
            return False

    def get_prompt(self, character_id: str) -> Optional[str]:
        """Get a character's prompt."""
        character_dir = self.characters_dir / character_id
        prompt_file = character_dir / "prompt.md"

        if not character_dir.exists():
            return None

        try:
            content = prompt_file.read_text(encoding='utf-8')

            # Replace {{daily}} placeholder with daily_edit.txt content
            if '{{daily}}' in content:
                daily_edit_path = Path(__file__).parent.parent.parent / "plugins" / "daily_note" / "daily_edit.txt"
                try:
                    daily_content = daily_edit_path.read_text(encoding='utf-8')
                    # Replace placeholders in daily_content
                    today = datetime.now().strftime('%Y-%m-%d')
                    daily_content = daily_content.replace('{TODAY}', today)
                    daily_content = daily_content.replace('{CHARACTER_ID}', character_id)
                    content = content.replace('{{daily}}', daily_content)
                    logger.info(f"[DailyNote] Replaced {{daily}} placeholder for {character_id} with today={today}")
                except Exception as e:
                    logger.warning(f"Failed to load daily_edit.txt: {e}")
                    content = content.replace('{{daily}}', '')

            return content
        except Exception as e:
            logger.error(f"Error reading prompt for {character_id}: {e}")
            return None

    def get_character_dir(self, character_id: str) -> Optional[Path]:
        """Get the directory path for a character."""
        character_dir = self.characters_dir / character_id
        if character_dir.exists():
            return character_dir
        return None

    def get_daily_dir(self, character_id: str) -> Optional[Path]:
        """Get the daily diary directory for a character (global data/daily/{name}/)."""
        character = self.get_character(character_id)
        if not character:
            return None
        return self.daily_dir / self._sanitize_name(character.name)

# Backward compatibility alias
CharacterStorageService = CharacterService
