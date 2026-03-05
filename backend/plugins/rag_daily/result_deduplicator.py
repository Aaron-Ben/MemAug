"""
Result Deduplicator Module for RAG Daily Plugin.

Implements intelligent result deduplication using SVD-based topic analysis
and residual pyramid projection for redundancy detection.

Uses residual selection algorithm to maximize diversity while maintaining relevance.
"""

import logging
from typing import Dict, List, Set, Optional, Any
import numpy as np

from .epa_module import EPAModule
from .residual_pyramid import ResidualPyramid


logger = logging.getLogger(__name__)


class ResultDeduplicator:

    def __init__(self, db, config: Optional[Dict[str, Any]] = None):
        """
        初始化主题提取器
        
        Args:
            db: 数据库连接/对象
            config: 配置字典，可选参数：
                - dimension: 向量维度 (默认 3072)
                - max_results: 最终保留的最大结果数 (默认 20)
                - topic_count: SVD 提取的主题数 (默认 8)
                - min_energy_ratio: 剩余能量阈值 (默认 0.1)
                - redundancy_threshold: 冗余阈值(余弦相似度) (默认 0.85)
        """
        # 初始化数据库连接
        self.db = db
        
        # 1. 初始化配置参数（设置默认值 + 合并用户配置）
        default_config = {
            'dimension': 3072,
            'max_results': 20,
            'topic_count': 8,
            'min_energy_ratio': 0.1,
            'redundancy_threshold': 0.85
        }

        if config is not None:
            default_config.update(config)

        # 最终配置赋值
        self.config = default_config
        
        # 2. 实例化 EPAModule（复用现有基础设施）
        self.epa = EPAModule(
            db=db,
            config={
                'dimension': self.config['dimension'],
                'max_basis_dim': self.config['topic_count'],
                'cluster_count': 16  # 针对结果集的小规模聚类
            }
        )
        
        # 3. 实例化残差金字塔计算器（用于投影计算）
        self.residual_calculator = ResidualPyramid(
            tag_index=None,
            db=db,
            config={
                'dimension': self.config['dimension']
            }
        )

    async def deduplicate(
        self,
        candidates: List[Dict],
        query_vector: np.ndarray,
    ) -> List[Dict]:
        """
        Deduplicate and select diverse results from candidates.

        Args:
            candidates: List of candidate results with 'vector' and 'score' fields
            query_vector: Original query vector (numpy array)

        Returns:
            Deduplicated and diverse list of results
        """
        if not candidates or len(candidates) == 0:
            return []

        # 1. Preprocess: filter results without vectors, ensure Float32Array
        valid_candidates = [
            c for c in candidates
            if c.get("vector") is not None or c.get("_vector") is not None
        ]

        if len(valid_candidates) <= 5:
            return candidates  # Too few results, no need to deduplicate

        logger.info(
            f"[ResultDeduplicator] Starting deduplication for "
            f"{len(valid_candidates)} candidates..."
        )

        # Extract vectors
        vectors = []
        for c in valid_candidates:
            v = c.get("vector") or c.get("_vector")
            if isinstance(v, np.ndarray):
                vectors.append(v.astype(np.float32))
            elif isinstance(v, list):
                vectors.append(np.array(v, dtype=np.float32))
            else:
                vectors.append(np.array(v, dtype=np.float32))

        # 2. SVD analysis on the current result set (no pre-trained Tag clusters)
        # Build a temporary clusterData structure for EPAModule
        cluster_data = {
            "vectors": vectors,
            "weights": [1.0] * len(vectors),  # Equal weights
            "labels": ["candidate"] * len(vectors),
        }

        # 3. Compute weighted PCA (SVD) to extract topic distribution of results
        # This tells us what aspects these search results mainly discuss
        svd_result = self.epa._compute_weighted_pca(cluster_data)
        topics = svd_result["U"]
        energies = svd_result["S"]

        # Filter out very weak topics
        significant_topics = []
        total_energy = float(np.sum(energies))
        cum_energy = 0.0

        for i in range(len(topics)):
            significant_topics.append(topics[i])
            cum_energy += float(energies[i])
            if cum_energy / total_energy > 0.95:
                break

        logger.info(
            f"[ResultDeduplicator] Identified {len(significant_topics)} "
            f"significant latent topics."
        )

        # 4. Residual Selection Algorithm
        # Goal: Select results that maximally cover query projections on significantTopics

        selected_indices: Set[int] = set()
        selected_results = []

        # 4.1 Prioritize keeping the most relevant to Query as #1 (Anchor)
        # Assume candidates are already sorted by score, take first
        # But for rigor, recalculate similarity with Query
        best_idx = -1
        best_sim = -1.0

        # Normalize Query
        n_query = self._normalize(query_vector)

        for i in range(len(vectors)):
            sim = self._dot_product(self._normalize(vectors[i]), n_query)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx != -1:
            selected_indices.add(best_idx)
            selected_results.append(valid_candidates[best_idx])

        # 4.2 Iterative selection: find candidates that best explain residual features
        max_rounds = self.max_results - 1

        # Initial orthogonal basis = [first place]
        current_basis = [vectors[best_idx]]

        for _ in range(max_rounds):
            max_projected_energy = -1.0
            next_best_idx = -1

            # Iterate through unselected candidates
            for i in range(len(vectors)):
                if i in selected_indices:
                    continue

                vec = vectors[i]

                # A. Calculate "difference" between this vector and selected set (residual)
                # Use ResidualPyramid's orthogonal projection logic
                # We want a vector with maximal component outside "selected basis"
                # (i.e., provides most new information)
                basis_dict_list = [{"vector": v.tobytes() if isinstance(v, np.ndarray) else v} for v in current_basis]
                projection_result = self.residual_calculator._compute_orthogonal_projection(vec, basis_dict_list)
                residual = np.array(projection_result.residual, dtype=np.float32)
                novelty_energy = self._magnitude(residual) ** 2

                # B. Meanwhile, this new info must be "relevant" new info, not noise
                # Check its projection on significantTopics
                # Simplified: as long as it has projection in Topics space,
                # and is unique relative to selected basis

                # Combined score: novelty * original relevance
                # Original score usually in candidates[i].score
                # If not, use sim calculated above
                original_score = valid_candidates[i].get("score", 0.5)
                score = novelty_energy * (original_score + 0.5)  # +0.5 smoothing

                if score > max_projected_energy:
                    max_projected_energy = score
                    next_best_idx = i

            if next_best_idx != -1:
                # Check if too similar (though residual projection already implies this,
                # explicit threshold is safer)
                # Actually, if residual magnitude is small, it means linearly correlated (similar)
                if max_projected_energy < 0.01:
                    logger.info(
                        "[ResultDeduplicator] Remaining candidates provide "
                        "negligible novelty. Stopping."
                    )
                    break

                selected_indices.add(next_best_idx)
                selected_results.append(valid_candidates[next_best_idx])
                current_basis.append(vectors[next_best_idx])
            else:
                break

        logger.info(
            f"[ResultDeduplicator] Selected {len(selected_results)} / "
            f"{len(valid_candidates)} diverse results."
        )

        return selected_results

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        mag = self._magnitude(vec)
        if mag > 1e-9:
            return vec / mag
        return vec.copy()

    def _dot_product(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute dot product of two vectors."""
        return float(np.dot(v1, v2))

    def _magnitude(self, vec: np.ndarray) -> float:
        """Calculate L2 magnitude of a vector."""
        return float(np.linalg.norm(vec))

