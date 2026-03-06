"""Character data models - Simplified for file system based storage.

Characters are stored as:
- data/characters/{uuid}/prompt.md - Character system prompt
- data/characters/{uuid}/daily/ - Diary files
- data/characters/{uuid}/chat/ - Chat history

Each character has:
- character_id: UUID4 (also used as directory name)
- name: Display name (from prompt.md first line)
- created_at: Creation timestamp (from directory mtime)
"""

from pydantic import BaseModel, Field
from typing import Optional


class UserCharacter(BaseModel):
    """User-created character stored in file system."""
    character_id: str = Field(..., description="UUID4 of the character")
    name: str = Field(..., description="Character display name")
    created_at: str = Field(..., description="ISO format creation timestamp")


# Empty class for backward compatibility - user preferences are not currently implemented
class UserCharacterPreference(BaseModel):
    """Placeholder for user character preferences (not currently implemented)."""
    pass
