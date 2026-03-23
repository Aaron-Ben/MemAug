"""V1 日记文件服务

从 app/services/diary/file_service.py 迁移
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from memory.v1.models import SessionLocal, DiaryFileTable
from app.services.character_service import CharacterService

logger = logging.getLogger(__name__)


# 默认目录（指向项目根目录的 data 文件夹）
DEFAULT_CHARACTERS_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "characters"
DEFAULT_DAILY_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "daily"


def get_characters_dir() -> Path:
    """获取角色目录"""
    path_str = os.getenv("CHARACTERS_DIR")
    if path_str:
        return Path(path_str)
    return DEFAULT_CHARACTERS_DIR


def get_daily_dir() -> Path:
    """获取全局日记目录"""
    path_str = os.getenv("DAILY_DIR")
    if path_str:
        return Path(path_str)
    return DEFAULT_DAILY_DIR


def calculate_file_checksum(file_path: Path) -> str:
    """计算文件的 MD5 哈希"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


class DiaryFileService:
    """基于文件系统的日记服务（只读操作）"""

    def __init__(self, characters_dir: Optional[Path] = None, daily_dir: Optional[Path] = None):
        self.characters_dir = characters_dir or get_characters_dir()
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        self.daily_dir = daily_dir or get_daily_dir()
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.character_service = CharacterService(self.characters_dir, self.daily_dir)

    def read_diary(self, path: str) -> Optional[Dict[str, any]]:
        """读取日记文件"""
        file_path = self.daily_dir / path
        if not file_path.exists():
            return None

        content = file_path.read_text(encoding='utf-8')
        stat = file_path.stat()
        mtime = int(stat.st_mtime)

        path_parts = path.split('/')
        name = path_parts[0] if path_parts else ""

        return {
            "path": path,
            "character_id": name,
            "content": content,
            "mtime": mtime
        }

    def list_diaries(self, character_id: str, limit: int = 10) -> List[Dict[str, any]]:
        """列出指定角色的日记文件"""
        character = self.character_service.get_character(character_id)
        if not character:
            return []

        name = self._sanitize_name(character.name)
        name_daily_dir = self.daily_dir / name

        db = SessionLocal()
        try:
            files = (db.query(DiaryFileTable)
                    .filter(DiaryFileTable.diary_name == name)
                    .order_by(DiaryFileTable.mtime.desc())
                    .limit(limit)
                    .all())

            result = []
            for file_record in files:
                file_path = self.daily_dir / file_record.path
                if file_path.exists():
                    content = file_path.read_text(encoding='utf-8')
                    result.append({
                        "path": file_record.path,
                        "character_id": character_id,
                        "content": content,
                        "mtime": file_record.mtime
                    })
            if result:
                return result

            if name_daily_dir.exists():
                for file_path in sorted(name_daily_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        relative_path = file_path.relative_to(self.daily_dir).as_posix()
                        stat = file_path.stat()
                        result.append({
                            "path": relative_path,
                            "character_id": character_id,
                            "content": content,
                            "mtime": int(stat.st_mtime)
                        })
                    except Exception:
                        continue

            return result

        finally:
            db.close()

    def _sanitize_name(self, name: str) -> str:
        """清理角色名称作为目录名"""
        import re
        sanitized = re.sub(r'[\\/:*?"<>|]', '', name.strip())
        sanitized = re.sub(r'\s+', '_', sanitized)
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized or 'unnamed'

    def list_all_diary_names(self) -> List[str]:
        """列出所有有日记的角色ID列表"""
        db = SessionLocal()
        try:
            names = db.query(DiaryFileTable.diary_name).distinct().all()
            return [name[0] for name in names]
        finally:
            db.close()

    def delete_diary(self, path: str) -> bool:
        """删除日记文件"""
        file_path = self.daily_dir / path
        db = SessionLocal()
        try:
            db.query(DiaryFileTable).filter(DiaryFileTable.path == path).delete()
            db.commit()

            if file_path.exists():
                file_path.unlink()
                logger.info(f"Diary file deleted: {path}")
                return True
            return False
        finally:
            db.close()

    def update_file_metadata(self, character_id: str) -> Dict[str, any]:
        """更新指定角色的日记文件元数据到数据库"""
        added_count = 0
        removed_count = 0
        updated_count = 0

        character = self.character_service.get_character(character_id)
        if not character:
            return {
                "character_id": character_id,
                "added": 0,
                "updated": 0,
                "removed": 0,
                "error": "Character not found"
            }

        name = self._sanitize_name(character.name)
        daily_dir = self.daily_dir / name

        db = SessionLocal()
        try:
            existing_files = {
                f.path: f
                for f in db.query(DiaryFileTable)
                       .filter(DiaryFileTable.diary_name == name)
                       .all()
            }

            if daily_dir.exists():
                for file_path in daily_dir.glob("*.txt"):
                    relative_path = file_path.relative_to(self.daily_dir).as_posix()
                    stat = file_path.stat()
                    mtime = int(stat.st_mtime)
                    size = stat.st_size
                    checksum = calculate_file_checksum(file_path)

                    if relative_path in existing_files:
                        record = existing_files[relative_path]
                        if (record.checksum != checksum or
                            record.mtime != mtime or
                            record.size != size):
                            record.checksum = checksum
                            record.mtime = mtime
                            record.size = size
                            record.updated_at = int(__import__('datetime').datetime.now().timestamp())
                            updated_count += 1
                        del existing_files[relative_path]
                    else:
                        new_record = DiaryFileTable(
                            path=relative_path,
                            diary_name=name,
                            checksum=checksum,
                            mtime=mtime,
                            size=size,
                            updated_at=int(__import__('datetime').datetime.now().timestamp())
                        )
                        db.add(new_record)
                        added_count += 1

            for path in existing_files:
                db.query(DiaryFileTable).filter(DiaryFileTable.path == path).delete()
                removed_count += 1

            db.commit()

            logger.info(f"Character {character_id} file metadata update completed: added={added_count}, updated={updated_count}, removed={removed_count}")

            return {
                "character_id": character_id,
                "added": added_count,
                "updated": updated_count,
                "removed": removed_count
            }
        finally:
            db.close()
