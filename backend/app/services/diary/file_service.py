"""File-based diary service for emotional companionship system.

Diaries are now stored in data/characters/{character_id}/daily/
- Each character has their own diary folder
- Database tracks file metadata (path, checksum, mtime, size)
- Supports creating, updating, and listing diary files
"""

import hashlib
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from app.models.database import SessionLocal, DiaryFileTable

logger = logging.getLogger(__name__)


# 默认角色目录
DEFAULT_CHARACTERS_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "characters"

# 旧的日记根目录 (保持兼容性)
DEFAULT_DIARY_ROOT = Path(__file__).parent.parent.parent.parent.parent / "data" / "diary"

# 忽略的文件夹列表
IGNORED_FOLDERS = ['MusicDiary']


def get_characters_dir() -> Path:
    """获取角色目录"""
    path_str = os.getenv("CHARACTERS_DIR")
    if path_str:
        return Path(path_str)
    return DEFAULT_CHARACTERS_DIR


def get_diary_root() -> Path:
    """获取旧的日记根目录 (已弃用)"""
    path_str = os.getenv("DIARY_ROOT_PATH")
    if path_str:
        return Path(path_str)
    return DEFAULT_DIARY_ROOT


def sanitize_path_component(name: str) -> str:
    """清理路径组件，确保安全"""
    if not name or isinstance(name, str):
        name = str(name) if name else "Untitled"

    sanitized = name \
        .replace('\\', '').replace('/', '').replace(':', '') \
        .replace('*', '').replace('?', '').replace('"', '') \
        .replace('<', '').replace('>', '').replace('|', '') \
        .replace('\x00', '').replace('\r', '').replace('\n', '') \
        .strip()

    # 限制长度
    return sanitized[:100] or "Untitled"


def calculate_file_checksum(file_path: Path) -> str:
    """计算文件的 MD5 哈希"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


class DiaryFileService:
    """基于文件系统的日记服务

    日记保存在 data/characters/{character_id}/daily/ 目录下
    """

    TAG_PATTERN = re.compile(r'^Tag:\s*(.+)$', re.MULTILINE | re.IGNORECASE)

    def __init__(self, characters_dir: Optional[Path] = None, character_id: Optional[str] = None):
        """初始化日记文件服务

        Args:
            characters_dir: 角色根目录，默认为 DEFAULT_CHARACTERS_DIR
            character_id: 可选的角色ID，用于限定操作范围
        """
        self.characters_dir = characters_dir or get_characters_dir()
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        self.character_id = character_id

    def _get_character_daily_dir(self, character_id: str) -> Path:
        """获取指定角色的日记目录

        Args:
            character_id: 角色 ID

        Returns:
            日记目录路径 data/characters/{character_id}/daily/
        """
        # 验证 character_id 是有效的 UUID
        try:
            uuid.UUID(character_id)
        except ValueError:
            # 如果不是 UUID，可能是旧的 sister_001 格式，直接使用
            pass

        daily_dir = self.characters_dir / character_id / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        return daily_dir

    def _ensure_tag_line(self, content: str, tag: Optional[str] = None) -> str:
        """确保内容有 Tag 行"""
        lines = content.split('\n')
        if lines:
            last_line = lines[-1].strip()
            if self.TAG_PATTERN.match(last_line):
                if tag and tag.strip():
                    return '\n'.join(lines[:-1]).rstrip() + f"\n\nTag: {tag.strip()}"
                return content

        if tag and tag.strip():
            return content.rstrip() + f"\n\nTag: {tag.strip()}"

        raise ValueError(
            "Tag is missing. Please provide a 'tag' parameter or add a 'Tag:' line at the end."
        )

    def get_file_path(self, character_id: str, date: str) -> Path:
        """获取日记文件路径

        Args:
            character_id: 角色 ID
            date: 日期 YYYY-MM-DD

        Returns:
            文件路径 data/characters/{character_id}/daily/{date}-{time}.txt
        """
        daily_dir = self._get_character_daily_dir(character_id)
        date_part = date.replace(".", "-").replace("/", "-").replace("\\", "-").strip()

        # 获取当前时间戳用于唯一文件名
        now = datetime.now()
        time_str = now.strftime("%H_%M_%S")
        filename = f"{date_part}-{time_str}.txt"

        return daily_dir / filename

    def create_diary(
        self,
        character_id: str,
        date: str,
        content: str,
        tag: Optional[str] = None
    ) -> Dict[str, any]:
        """创建日记文件

        Args:
            character_id: 角色 ID
            date: 日期 YYYY-MM-DD
            content: 日记内容
            tag: 可选标签

        Returns:
            包含文件元数据的字典
        """
        # 确保 Tag 行存在
        content_with_tag = self._ensure_tag_line(content, tag)

        # 获取文件路径
        file_path = self.get_file_path(character_id, date)

        # 如果文件已存在，添加计数器后缀
        counter = 1
        original_path = file_path
        while file_path.exists():
            stem = original_path.stem
            extension = original_path.suffix
            file_path = original_path.parent / f"{stem}({counter}){extension}"
            counter += 1

        # 写入文件
        file_path.write_text(content_with_tag, encoding='utf-8')

        # 获取文件元数据
        stat = file_path.stat()
        mtime = int(stat.st_mtime)
        size = stat.st_size
        checksum = calculate_file_checksum(file_path)

        # 计算相对路径 (从 characters 目录开始)
        relative_path = file_path.relative_to(self.characters_dir).as_posix()

        # 保存到数据库
        db = SessionLocal()
        try:
            file_record = DiaryFileTable(
                path=relative_path,
                diary_name=character_id,
                checksum=checksum,
                mtime=mtime,
                size=size,
                updated_at=int(datetime.now().timestamp())
            )
            db.add(file_record)
            db.commit()
            db.refresh(file_record)

            logger.info(f"Diary file created: {relative_path}")

            return {
                "status": "success",
                "message": f"Diary saved to {relative_path}",
                "data": {
                    "id": file_record.id,
                    "path": relative_path,
                    "character_id": character_id,
                    "content": content_with_tag,
                    "mtime": mtime,
                    "size": size
                }
            }
        finally:
            db.close()

    def read_diary(self, path: str) -> Optional[Dict[str, any]]:
        """读取日记文件

        Args:
            path: 文件相对路径

        Returns:
            包含文件内容和元数据的字典，如果文件不存在返回 None
        """
        file_path = self.characters_dir / path

        if not file_path.exists():
            return None

        content = file_path.read_text(encoding='utf-8')
        stat = file_path.stat()
        mtime = int(stat.st_mtime)

        # 从路径中提取 character_id (第一个路径组件)
        path_parts = path.split('/')
        character_id = path_parts[0] if path_parts else ""

        return {
            "path": path,
            "character_id": character_id,
            "content": content,
            "mtime": mtime
        }

    def update_diary(
        self,
        target: str,
        replace: str,
        character_id: Optional[str] = None
    ) -> Dict[str, any]:
        """更新日记文件（查找并替换内容）

        Args:
            target: 要查找的旧内容（至少15字符）
            replace: 替换的新内容
            character_id: 可选的角色ID，用于限定搜索范围

        Returns:
            操作结果
        """
        if len(target) < 15:
            return {
                "status": "error",
                "message": f"Security check failed: 'target' must be at least 15 characters. Provided: {len(target)}"
            }

        db = SessionLocal()
        try:
            # 构建查询
            query = db.query(DiaryFileTable)
            if character_id:
                query = query.filter(DiaryFileTable.diary_name == character_id)

            # 按 mtime 降序排列（最新的在前）
            files = query.order_by(DiaryFileTable.mtime.desc()).all()

            for file_record in files:
                file_path = self.characters_dir / file_record.path
                if not file_path.exists():
                    continue

                content = file_path.read_text(encoding='utf-8')

                if target in content:
                    # 替换内容
                    new_content = content.replace(target, replace, 1)
                    file_path.write_text(new_content, encoding='utf-8')

                    # 更新元数据
                    stat = file_path.stat()
                    file_record.mtime = int(stat.st_mtime)
                    file_record.size = stat.st_size
                    file_record.checksum = calculate_file_checksum(file_path)
                    file_record.updated_at = int(datetime.now().timestamp())
                    db.commit()

                    logger.info(f"Diary file updated: {file_record.path}")

                    return {
                        "status": "success",
                        "message": f"Successfully edited diary: {file_record.path}",
                        "path": file_record.path
                    }

            char_msg = f" for character '{character_id}'" if character_id else ""
            return {
                "status": "error",
                "message": f"Target content not found in any diary{char_msg}."
            }
        finally:
            db.close()

    def list_diaries(self, character_id: str, limit: int = 10) -> List[Dict[str, any]]:
        """列出指定角色的日记文件

        Args:
            character_id: 角色 ID
            limit: 返回数量限制

        Returns:
            日记文件列表
        """
        db = SessionLocal()
        try:
            files = (db.query(DiaryFileTable)
                    .filter(DiaryFileTable.diary_name == character_id)
                    .order_by(DiaryFileTable.mtime.desc())
                    .limit(limit)
                    .all())

            result = []
            for file_record in files:
                file_path = self.characters_dir / file_record.path
                if file_path.exists():
                    content = file_path.read_text(encoding='utf-8')
                    result.append({
                        "path": file_record.path,
                        "character_id": character_id,
                        "content": content,
                        "mtime": file_record.mtime
                    })

            return result
        finally:
            db.close()

    def list_all_diary_names(self) -> List[str]:
        """列出所有有日记的角色ID列表

        Returns:
            角色 ID 列表
        """
        db = SessionLocal()
        try:
            names = db.query(DiaryFileTable.diary_name).distinct().all()
            return [name[0] for name in names if name[0] not in IGNORED_FOLDERS]
        finally:
            db.close()

    def delete_diary(self, path: str) -> bool:
        """删除日记文件

        Args:
            path: 文件相对路径

        Returns:
            是否删除成功
        """
        file_path = self.characters_dir / path

        db = SessionLocal()
        try:
            # 删除数据库记录
            db.query(DiaryFileTable).filter(DiaryFileTable.path == path).delete()
            db.commit()

            # 删除文件
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Diary file deleted: {path}")
                return True

            return False
        finally:
            db.close()

    def sync_character_diaries(self, character_id: str) -> Dict[str, any]:
        """同步指定角色的日记文件到数据库

        扫描角色的日记目录，添加新文件到数据库，删除不存在的文件记录

        Args:
            character_id: 角色 ID

        Returns:
            同步结果统计
        """
        added_count = 0
        removed_count = 0
        updated_count = 0

        daily_dir = self._get_character_daily_dir(character_id)

        db = SessionLocal()
        try:
            # 获取该角色的所有文件
            existing_files = {
                f.path: f
                for f in db.query(DiaryFileTable)
                       .filter(DiaryFileTable.diary_name == character_id)
                       .all()
            }

            # 扫描文件系统
            if daily_dir.exists():
                for file_path in daily_dir.glob("*.txt"):
                    relative_path = file_path.relative_to(self.characters_dir).as_posix()
                    stat = file_path.stat()
                    mtime = int(stat.st_mtime)
                    size = stat.st_size
                    checksum = calculate_file_checksum(file_path)

                    if relative_path in existing_files:
                        # 更新现有记录
                        record = existing_files[relative_path]
                        if (record.checksum != checksum or
                            record.mtime != mtime or
                            record.size != size):
                            record.checksum = checksum
                            record.mtime = mtime
                            record.size = size
                            record.updated_at = int(datetime.now().timestamp())
                            updated_count += 1
                        del existing_files[relative_path]
                    else:
                        # 添加新记录
                        new_record = DiaryFileTable(
                            path=relative_path,
                            diary_name=character_id,
                            checksum=checksum,
                            mtime=mtime,
                            size=size,
                            updated_at=int(datetime.now().timestamp())
                        )
                        db.add(new_record)
                        added_count += 1

            # 删除数据库中不存在于文件系统的记录
            for path in existing_files:
                db.query(DiaryFileTable).filter(DiaryFileTable.path == path).delete()
                removed_count += 1

            db.commit()

            logger.info(f"Character {character_id} diary sync completed: added={added_count}, updated={updated_count}, removed={removed_count}")

            return {
                "character_id": character_id,
                "added": added_count,
                "updated": updated_count,
                "removed": removed_count
            }
        finally:
            db.close()
