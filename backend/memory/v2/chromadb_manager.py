"""ChromaDB manager for vector storage and search."""
import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from pathlib import Path

logger = logging.getLogger(__name__)

# Default ChromaDB persistence directory
DEFAULT_PERSIST_DIR = Path(__file__).parent.parent.parent / "data" / "chroma"


class ChromaDBManager:
    """ChromaDB manager for memory vector storage."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "memories",
    ):
        """Initialize ChromaDB manager.

        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection to use
        """
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings()
        )

        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            matadata={"hnsw:space":"cosine"}
        )


    async def search_similar_memories(
        self,
        owner_space: Optional[str],
        category_uri_prefix: str,
        query_vector: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for similar memories.

        Args:
            owner_space: Owner space filter (e.g., user or agent name)
            category_uri_prefix: Category URI prefix filter
            query_vector: Query embedding vector
            limit: Maximum number of results

        Returns:
            List of memory dicts with uri, abstract, content, etc.
        """

        where_clause = {
            "$and": [
                {"context_type": {"$eq": "memory"}},
                {"level": {"$eq": 2}},
            ]
        }

        if owner_space:
            where_clause["$and"].append({"owner_space": {"$eq": owner_space}})

        # uri 前缀匹配（ChromaDB 支持 $startsWith）
        if category_uri_prefix:
            where_clause["$and"].append(
                {"uri": {"$startsWith": category_uri_prefix}}
            )

        # 执行向量搜索
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=where_clause,
            include=["metadatas", "distances", "documents"]
        )

        similar_memories = []
        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0

                # ChromaDB 返回的是距离（越小越相似），需要转换为相似度分数
                # cosine 距离范围 [0, 2]，转换为相似度 [1, -1]，再归一化到 [0, 1]
                score = 1 - (distance / 2)  # 或者直接用 1 - distance

                memory_record = {
                    "id": memory_id,
                    "_score": score,
                    **metadata,
                }
                similar_memories.append(memory_record)

        return similar_memories
    