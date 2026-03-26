"""ChromaDB manager for vector storage and search."""
import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from pathlib import Path

logger = logging.getLogger(__name__)

# Default ChromaDB persistence directory - project root
DEFAULT_PERSIST_DIR = Path(__file__).parent.parent.parent.parent / "chroma-db"


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
        # Use default if not provided
        if persist_dir is None:
            persist_dir = str(DEFAULT_PERSIST_DIR)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings()
        )

        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
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

                # 后处理过滤 uri 前缀
                if category_uri_prefix and not metadata.get("uri", "").startswith(category_uri_prefix):
                    continue

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

    async def add_memory(
        self,
        memory_id: str,
        embedding: List[float],
        text: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """Add a memory to ChromaDB.

        Args:
            memory_id: Unique ID for the memory
            embedding: Vector embedding
            text: Text content to store
            metadata: Metadata dict (uri, category, etc.)

        Returns:
            True if successful
        """
        try:
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
            )
            logger.info(f"Added memory to ChromaDB: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add memory to ChromaDB: {e}")
            return False

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from ChromaDB.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory from ChromaDB: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory from ChromaDB: {e}")
            return False
    