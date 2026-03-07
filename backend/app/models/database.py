"""Database configuration and models for emotional companionship system."""

import os
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# 使用 SQLite 数据库
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./../VectorStore/emotional_companionship.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DiaryFileTable(Base):
    """日记表"""
    __tablename__ = "diary_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(Text, unique=True, nullable=False, index=True)
    diary_name = Column(String, nullable=False, index=True)
    checksum = Column(String, nullable=False)
    mtime = Column(Integer, nullable=False)
    size = Column(Integer, nullable=False)
    updated_at = Column(Integer, nullable=False)

    # Relationships
    chunks = relationship("ChunkTable", back_populates="file", cascade="all, delete-orphan")


class ChunkTable(Base):
    """文本分块表 - 存储日记内容的分块和向量"""
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey("diary_files.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)  # 分块在文档中的顺序
    content = Column(Text, nullable=False)  # 分块内容
    vector = Column(Text, nullable=True)  # 向量（JSON 数组格式）

    # Relationships
    file = relationship("DiaryFileTable", back_populates="chunks")

    # Unique constraint and index
    __table_args__ = (
        UniqueConstraint("file_id", "chunk_index", name="unique_file_chunk"),
        Index("idx_file_id_chunk_index", "file_id", "chunk_index"),
    )


class TagTable(Base):
    """标签表 - 存储标签和向量"""
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False, index=True)  # 标签名称
    vector = Column(Text, nullable=True)  # 向量（JSON 数组格式）

    # Relationships
    files = relationship("FileTagTable", back_populates="tag")


class FileTagTable(Base):
    """文件-标签关联表 - 多对多关系"""
    __tablename__ = "file_tags"

    file_id = Column(Integer, ForeignKey("diary_files.id", ondelete="CASCADE"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)

    # Relationships
    file = relationship("DiaryFileTable")
    tag = relationship("TagTable", back_populates="files")


class KVStoreTable(Base):
    """通用键值存储表"""
    __tablename__ = "kv_store"

    key = Column(String, primary_key=True, nullable=False)
    value = Column(Text, nullable=True)  # 值（可以是 JSON 字符串）
    vector = Column(Text, nullable=True)  # 可选的向量（JSON 数组格式）
    updated_at = Column(Integer, nullable=False)  # 更新时间戳


def init_db():
    """初始化数据库，创建所有表"""
    Base.metadata.create_all(bind=engine)
