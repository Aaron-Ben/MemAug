"""Diary-related services for emotional companionship system.

File-based diary system:
- Diaries stored as text files in data/characters/{character_id}/daily/
- Database tracks file metadata (path, checksum, mtime, size)
- Creating and updating diaries is handled by the DailyNote plugin
"""

from app.services.diary.file_service import DiaryFileService

__all__ = [
    "DiaryFileService",
]
