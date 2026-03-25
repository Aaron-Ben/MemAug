"""
Hierarchical retriever for V2 memory system.

基于 ChromaDB 向量检索和文件系统读取的层级记忆检索实现。
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

logger = logging.getLogger(__name__)

# 数据目录基础路径
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


class ContextType(str, Enum):
    """Context type for retrieval."""
    MEMORY = "memory"
    SESSION = "session"


class SpaceType(str, Enum):
    """Space type for retrieval."""
    USER = "user"
    AGENT = "agent"


@dataclass
class RelatedContext:
    """Related context with summary."""
    uri: str
    abstract: str


@dataclass
class MatchedContext:
    """Matched context from retrieval."""
    uri: str
    context_type: ContextType
    level: int = 2
    abstract: str = ""
    overview: Optional[str] = None
    category: str = ""
    score: float = 0.0
    match_reason: str = ""
    relations: List[RelatedContext] = field(default_factory=list)


@dataclass
class QueryResult:
    """Result for a query."""
    query: str
    matched_contexts: List[MatchedContext]
    searched_directories: List[str]


class HierarchicalRetriever:
    """层级记忆检索器，支持 user 和 agent 空间。"""

    def __init__(
        self,
        chromadb_manager: Any,
        embedding_service: Any,
        data_dir: Optional[Path] = None,
    ):
        """初始化 HierarchicalRetriever.

        Args:
            chromadb_manager: ChromaDBManager 实例
            embedding_service: EmbeddingService 实例
            data_dir: 数据目录路径，默认为项目 data 目录
        """
        self._chromadb = chromadb_manager
        self._embedding = embedding_service
        self._data_dir = data_dir or DATA_DIR
        self._fs_cache: Dict[str, Any] = {}

    async def retrieve(
        self,
        query: str,
        user: str,
        space: SpaceType = SpaceType.USER,
        limit: int = 5,
    ) -> QueryResult:
        """检索记忆.

        Args:
            query: 查询文本
            user: 用户/代理标识
            space: 空间类型 (USER 或 AGENT)
            limit: 返回结果数量限制

        Returns:
            QueryResult: 包含 matched_contexts 和 searched_directories
        """
        searched_dirs: List[str] = []

        # Step 1: 向量化查询
        query_vector = await self._embedding.get_single_embedding(query)
        if not query_vector:
            logger.warning("[HierarchicalRetriever] Failed to embed query")
            return QueryResult(
                query=query,
                matched_contexts=[],
                searched_directories=searched_dirs,
            )

        results: List[MatchedContext] = []

        # Step 2: ChromaDB 向量检索 (L2 detail level)
        chroma_results = await self._search_detail_level(
            query_vector=query_vector,
            user=user,
            space=space,
            limit=limit,
        )
        results.extend(chroma_results)
        searched_dirs.extend(self._get_search_directories(user, space))

        # Step 3: 文件系统检索 (L0/L1 - abstract/overview)
        fs_results = await self._search_fs_levels(
            query=query,
            user=user,
            space=space,
            limit=limit,
        )
        results.extend(fs_results)

        # Step 4: 按 score 排序并返回
        results.sort(key=lambda x: x.score, reverse=True)
        final_results = results[:limit]

        logger.info(
            f"[HierarchicalRetriever] Retrieved {len(final_results)} results for query: {query[:50]}..."
        )

        return QueryResult(
            query=query,
            matched_contexts=final_results,
            searched_directories=searched_dirs,
        )

    async def _search_detail_level(
        self,
        query_vector: List[float],
        user: str,
        space: SpaceType,
        limit: int,
    ) -> List[MatchedContext]:
        """从 ChromaDB 检索 L2 详细内容."""
        results: List[MatchedContext] = []

        # 构建 owner_space 和 category_uri_prefix
        if space == SpaceType.AGENT:
            owner_space = f"agent:{user}"
        else:
            owner_space = user

        category_prefix = f"data/{space.value}/{user}/memories/"

        try:
            chroma_results = await self._chromadb.search_similar_memories(
                owner_space=owner_space,
                category_uri_prefix=category_prefix,
                query_vector=query_vector,
                limit=limit,
            )

            for r in chroma_results:
                results.append(
                    MatchedContext(
                        uri=r.get("uri", ""),
                        context_type=ContextType.MEMORY,
                        level=r.get("level", 2),
                        abstract=r.get("abstract", ""),
                        overview=r.get("overview", None),
                        category=r.get("category", ""),
                        score=r.get("_score", 0.0),
                    )
                )
        except Exception as e:
            logger.error(f"[HierarchicalRetriever] ChromaDB search failed: {e}")

        return results

    async def _search_fs_levels(
        self,
        query: str,
        user: str,
        space: SpaceType,
        limit: int,
    ) -> List[MatchedContext]:
        """从文件系统检索 L0 (abstract) 和 L1 (overview)."""
        results: List[MatchedContext] = []

        # 搜索会话文件
        session_results = await self._search_session_files(query, user, limit)
        results.extend(session_results)

        # 搜索记忆文件
        memory_results = await self._search_memory_files(query, user, space, limit)
        results.extend(memory_results)

        return results

    async def _search_session_files(
        self,
        query: str,
        user: str,
        limit: int,
    ) -> List[MatchedContext]:
        """搜索会话目录中的 .abstract.md 和 .overview.md 文件."""
        results: List[MatchedContext] = []
        session_dir = self._data_dir / "session" / user

        if not session_dir.exists():
            return results

        try:
            # 遍历所有会话目录
            for session_path in session_dir.iterdir():
                if not session_path.is_dir():
                    continue

                session_id = session_path.name

                # 读取 .abstract.md (L0)
                abstract_file = session_path / ".abstract.md"
                if abstract_file.exists():
                    content = await self._read_file(abstract_file)
                    if content:
                        score = self._calculate_text_similarity(query, content)
                        results.append(
                            MatchedContext(
                                uri=f"session/{session_id}/.abstract.md",
                                context_type=ContextType.SESSION,
                                level=0,
                                abstract=content,
                                category="session",
                                score=score,
                            )
                        )

                # 读取 .overview.md (L1)
                overview_file = session_path / ".overview.md"
                if overview_file.exists():
                    content = await self._read_file(overview_file)
                    if content:
                        score = self._calculate_text_similarity(query, content)
                        results.append(
                            MatchedContext(
                                uri=f"session/{session_id}/.overview.md",
                                context_type=ContextType.SESSION,
                                level=1,
                                overview=content,
                                category="session",
                                score=score,
                            )
                        )

                # 读取历史归档
                history_dir = session_path / "history"
                if history_dir.exists():
                    for archive_dir in history_dir.iterdir():
                        if not archive_dir.is_dir():
                            continue

                        # archive_X .abstract.md
                        archive_abstract = archive_dir / ".abstract.md"
                        if archive_abstract.exists():
                            content = await self._read_file(archive_abstract)
                            if content:
                                score = self._calculate_text_similarity(query, content)
                                results.append(
                                    MatchedContext(
                                        uri=f"session/{session_id}/history/{archive_dir.name}/.abstract.md",
                                        context_type=ContextType.SESSION,
                                        level=0,
                                        abstract=content,
                                        category="session_archive",
                                        score=score,
                                    )
                                )

                        # archive_X .overview.md
                        archive_overview = archive_dir / ".overview.md"
                        if archive_overview.exists():
                            content = await self._read_file(archive_overview)
                            if content:
                                score = self._calculate_text_similarity(query, content)
                                results.append(
                                    MatchedContext(
                                        uri=f"session/{session_id}/history/{archive_dir.name}/.overview.md",
                                        context_type=ContextType.SESSION,
                                        level=1,
                                        overview=content,
                                        category="session_archive",
                                        score=score,
                                    )
                                )
        except Exception as e:
            logger.error(f"[HierarchicalRetriever] Failed to search session files: {e}")

        # 按 score 排序，取 top N
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def _search_memory_files(
        self,
        query: str,
        user: str,
        space: SpaceType,
        limit: int,
    ) -> List[MatchedContext]:
        """搜索记忆目录中的记忆文件."""
        results: List[MatchedContext] = []
        memories_dir = self._data_dir / space.value / user / "memories"

        if not memories_dir.exists():
            return results

        try:
            # 遍历各分类目录
            for category_dir in memories_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                category = category_dir.name

                # 读取该分类下的所有 .md 文件
                for mem_file in category_dir.glob("*.md"):
                    if mem_file.name.startswith("."):
                        continue

                    content = await self._read_file(mem_file)
                    if not content:
                        continue

                    score = self._calculate_text_similarity(query, content)
                    if score > 0.05:  # 简单阈值过滤
                        # 提取 abstract (取第一段)
                        abstract = self._extract_abstract(content)

                        results.append(
                            MatchedContext(
                                uri=f"{space.value}/{user}/memories/{category}/{mem_file.name}",
                                context_type=ContextType.MEMORY,
                                level=0,
                                abstract=abstract,
                                category=category,
                                score=score,
                            )
                        )
        except Exception as e:
            logger.error(f"[HierarchicalRetriever] Failed to search memory files: {e}")

        # 按 score 排序，取 top N
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def _read_file(self, file_path: Path) -> str:
        """异步读取文件内容."""
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                return await f.read()
        except Exception as e:
            logger.warning(f"[HierarchicalRetriever] Failed to read {file_path}: {e}")
            return ""

    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """基于关键词匹配的相似度计算，支持中英文."""
        if not query or not text:
            return 0.0

        # 转换为小写
        query_lower = query.lower()
        text_lower = text.lower()

        # 中文：检查每个字符是否出现在文本中
        # 英文：按空格分割
        if " " in query_lower:
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
        else:
            # 中文字符匹配：每个查询字符都在文本中就算匹配
            query_chars = set(query_lower)
            text_chars = set(text_lower)
            query_words = query_chars
            text_words = text_chars

        if not query_words:
            return 0.0

        intersection = query_words & text_words
        return len(intersection) / len(query_words)

    def _extract_abstract(self, content: str) -> str:
        """从完整内容提取 abstract (取第一段或前 200 字符)."""
        if not content:
            return ""

        lines = content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:200]

        return content[:200]

    def _get_search_directories(self, user: str, space: SpaceType) -> List[str]:
        """获取检索的目录列表."""
        return [
            f"data/{space.value}/{user}/memories",
            f"data/session/{user}",
        ]


__all__ = [
    "HierarchicalRetriever",
    "MatchedContext",
    "RelatedContext",
    "QueryResult",
    "ContextType",
    "SpaceType",
]