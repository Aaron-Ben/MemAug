"""Diary data models for emotional companionship system.

Diaries are stored in data/characters/{character_id}/daily/
"""

from pydantic import BaseModel, Field


class DiaryEntry(BaseModel):
    """日记条目数据模型"""
    path: str = Field(..., description="文件相对路径 (从 characters 目录开始)")
    character_id: str = Field(..., description="角色 ID")
    content: str = Field(..., description="日记内容（第一人称，包含末尾的Tag行）")
    mtime: int = Field(..., description="文件修改时间戳")
