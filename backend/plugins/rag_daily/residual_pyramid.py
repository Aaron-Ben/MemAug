"""
Residual Pyramid Module for RAG Daily Plugin.

Implements multi-level semantic residual analysis using Gram-Schmidt orthogonalization.
Provides hierarchical vector decomposition for enhanced semantic retrieval.

uses Rust vector_db for high-performance operations.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class PyramidLevel:
    """A single level in the residual pyramid."""

    def __init__(
        self,
        level: int,
        tags: List[Dict],
        projection_magnitude: float,
        residual_magnitude: float,
        residual_energy_ratio: float,
        energy_explained: float,
        handshake_features: Optional[Dict] = None,
    ):
        self.level = level
        self.tags = tags  # List of tag info with similarity, contribution, handshake
        self.projection_magnitude = projection_magnitude
        self.residual_magnitude = residual_magnitude
        self.residual_energy_ratio = residual_energy_ratio
        self.energy_explained = energy_explained
        self.handshake_features = handshake_features

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "tags": self.tags,
            "projection_magnitude": float(self.projection_magnitude),
            "residual_magnitude": float(self.residual_magnitude),
            "residual_energy_ratio": float(self.residual_energy_ratio),
            "energy_explained": float(self.energy_explained),
            "handshake_features": self.handshake_features,
        }


class ResidualPyramid:
    """
    Residual Pyramid for multi-level semantic analysis.

    Features:
    - Gram-Schmidt orthogonal projection using Rust vector_db
    - Multi-level residual decomposition with iterative search
    - Handshake feature extraction
    - Energy-based level selection
    """

    def __init__(self, tag_index, db, config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the residual pyramid.

        Args:
            tag_index: VexusIndex instance for vector search and Rust operations
            db: Database connection for fetching tag details
            config: Configuration dictionary
        """
        self.tag_index = tag_index
        self.db = db
        default_config = {
            'max_levels': 3,
            'top_k': 10,
            'min_energy_ratio': 0.1,
            'dimension': 3072
        }
        if config is not None:
            default_config.update(config)

        # 最终配置
        self.config = default_config

        # Initialize instance attributes
        self._dim = self.config['dimension']
        self.levels: List[PyramidLevel] = []
        self.total_explained_energy: float = 0.0
        self.final_residual: Optional[np.ndarray] = None
        self.features: Dict[str, Any] = {}

    def analyze(self, query_vector: np.ndarray, tags: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Analyze a query vector using the residual pyramid.

        Iteratively searches for tags at each level using the current residual,
        then computes orthogonal projection to find the next residual.

        Args:
            query_vector: Input query vector (numpy array or list)
            tags: Optional list of tags with vectors. If provided, uses these directly
                  instead of searching. Each tag should be a dict with 'id', 'name',
                  and 'vector' keys.

        Returns:
            Analysis results dictionary with levels, energy, and features
        """
        logger.info(f"[ResidualPyramid] 🔺 Starting analysis (max_levels={self.config['max_levels']}, top_k={self.config['top_k']})")

        self.levels.clear()
        self.total_explained_energy = 0.0

        # Convert to numpy array and ensure float32
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)
        elif query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        # Calculate original energy
        original_magnitude = self._magnitude(query_vector)
        original_energy = original_magnitude ** 2

        if original_energy < 1e-12:
            return self._empty_result()

        current_residual = query_vector.copy()

        for level in range(self.config["max_levels"]):
            # 1. Get tags - either from provided list or search
            if tags is not None and level == 0:
                # Use provided tags directly (only for first level)
                raw_tags = tags[:self.config["top_k"]]
            else:
                # Search for tags using current residual
                residual_bytes = self._vector_to_bytes(current_residual)
                tag_results = self._search_tags(residual_bytes)

                if not tag_results:
                    logger.debug(f"[ResidualPyramid] No tags found at level {level}")
                    break

                # Get tag vectors from database
                tag_ids = [r.id for r in tag_results]
                raw_tags = self._get_tag_vectors(tag_ids)

            if not raw_tags:
                break

            # 3. Compute orthogonal projection using Rust
            projection_result = self._compute_orthogonal_projection(
                current_residual, raw_tags
            )

            # 4. Calculate energy metrics
            residual_magnitude = self._magnitude(
                np.array(projection_result.residual, dtype=np.float32)
            )
            residual_energy = residual_magnitude ** 2
            current_energy = self._magnitude(current_residual) ** 2

            energy_explained_by_level = max(0, current_energy - residual_energy) / original_energy

            # 5. Compute handshake features
            handshake_result = self._compute_handshakes(current_residual, raw_tags)
            handshake_features = self._analyze_handshakes(handshake_result)

            # 6. Build tag info list
            tag_info_list = []
            for i, tag in enumerate(raw_tags):
                # Find corresponding search result if available
                if tags is not None and level == 0:
                    # Using provided tags - compute similarity manually
                    tag_vec = np.frombuffer(tag["vector"], dtype=np.float32)
                    similarity = float(np.dot(
                        current_residual.astype(np.float64),
                        tag_vec.astype(np.float64)
                    ) / (
                        np.linalg.norm(current_residual) * np.linalg.norm(tag_vec) + 1e-9
                    ))
                else:
                    # Using search results
                    search_res = next((r for r in tag_results if r.id == tag["id"]), None)
                    similarity = search_res.score if search_res else 0.0

                contribution = projection_result.basis_coefficients[i] if i < len(projection_result.basis_coefficients) else 0.0
                handshake_mag = handshake_result.magnitudes[i] if i < len(handshake_result.magnitudes) else 0.0

                tag_info_list.append({
                    "id": tag["id"],
                    "name": tag.get("name", ""),
                    "similarity": float(similarity),
                    "contribution": float(contribution),
                    "handshake_magnitude": float(handshake_mag),
                })

            # 7. Create pyramid level
            pyramid_level = PyramidLevel(
                level=level,
                tags=tag_info_list,
                projection_magnitude=self._magnitude(
                    np.array(projection_result.projection, dtype=np.float32)
                ),
                residual_magnitude=residual_magnitude,
                residual_energy_ratio=residual_energy / original_energy,
                energy_explained=energy_explained_by_level,
                handshake_features=handshake_features,
            )
            self.levels.append(pyramid_level)
            self.total_explained_energy += energy_explained_by_level

            # Update residual for next iteration
            current_residual = np.array(projection_result.residual, dtype=np.float32)

            # 8. Check energy threshold
            if (residual_energy / original_energy) < self.config["min_energy_ratio"]:
                logger.debug(
                    f"[ResidualPyramid] Stopped at level {level}, "
                    f"energy ratio: {residual_energy / original_energy:.3f}"
                )
                break

            logger.info(f"[ResidualPyramid] 🔻 Level {level}: {len(tag_info_list)} tags, energy_explained={energy_explained_by_level:.2%}")

        self.final_residual = current_residual
        self.features = self._extract_pyramid_features()

        logger.info(f"[ResidualPyramid] ✅ Analysis complete: {len(self.levels)} levels, total_energy={self.total_explained_energy:.2%}")

        return self._compile_results()

    def _search_tags(self, query_bytes: bytes) -> List[Any]:
        """Search for tags using the VexusIndex."""
        if self.tag_index is None:
            return []

        try:
            return self.tag_index.search(query_bytes, self.config["top_k"])
        except Exception as e:
            logger.warning(f"[ResidualPyramid] Search failed: {e}")
            return []

    def _get_tag_vectors(self, tag_ids: List[int]) -> List[Dict]:
        """Fetch tag vectors from the database."""
        if self.db is None:
            return []

        placeholders = ",".join(["?" for _ in tag_ids])
        query = f"SELECT id, name, vector FROM tags WHERE id IN ({placeholders})"

        try:
            cursor = self.db.execute(query, tag_ids)
            rows = cursor.fetchall()

            result = []
            for row in rows:
                tag_id, name, vector_blob = row
                if vector_blob:
                    result.append({
                        "id": tag_id,
                        "name": name,
                        "vector": vector_blob,
                    })
            return result
        except Exception as e:
            logger.warning(f"[ResidualPyramid] Failed to fetch tags: {e}")
            return []

    def _compute_orthogonal_projection(
        self, vector: np.ndarray, tags: List[Dict]
    ) -> Any:
        """
        Compute orthogonal projection using Rust vector_db.

        Uses Gram-Schmidt orthogonalization to project the vector
        onto the subspace spanned by the tag vectors.
        """
        if self.tag_index is None:
            # Fallback to Python implementation
            raise RuntimeError("tag_index not initiallized")
        flattened_tags = self._flatten_tag_vectors(tags)

        vector_bytes = self._vector_to_bytes(vector)

        result = self.tag_index.compute_orthogonal_projection(
            vector_bytes, flattened_tags, len(tags)
        )
        return result


    def _compute_handshakes(
        self, vector: np.ndarray, tags: List[Dict]
    ) -> Any:
        """
        Compute handshake features using Rust vector_db.

        Handshakes measure the directional difference between the query
        and each tag vector.
        """
        if self.tag_index is None:
            raise RuntimeError("tag_index not initiallized")

        flattened_tags = self._flatten_tag_vectors(tags)
        vector_bytes = self._vector_to_bytes(vector)

        result = self.tag_index.compute_handshakes(
            vector_bytes, flattened_tags, len(tags)
        )

        # 结构化 directions: 扁平数组 -> 每个tag一个方向向量
        dim = self._dim
        n = len(tags)
        structured_directions = []

        for i in range(n):
            start = i * dim
            end = start + dim
            structured_directions.append(result.directions[start:end])

        # 保留7位小数
        rounded_magnitudes = [round(m, 7) for m in result.magnitudes]
        rounded_directions = [
            [round(x, 7) for x in dir_vec]
            for dir_vec in structured_directions
        ]

        # 包装成新对象返回
        class StructuredHandshakeResult:
            def __init__(self, magnitudes, directions):
                self.magnitudes = magnitudes
                self.directions = directions

        return StructuredHandshakeResult(rounded_magnitudes, rounded_directions)

    def _analyze_handshakes(self, handshake_result: Any) -> Dict:
        """
        Analyze handshake features to extract semantic patterns.

        Computes:
        - direction_coherence: Consistency of deviation directions
        - pattern_strength: Similarity between tag deviations
        - novelty_signal: How novel the query is
        - noise_signal: How random the deviations are
        """
        n = len(handshake_result.magnitudes)
        if n == 0:
            return None

        # directions 已经是结构化的 [[x,y,z], [x,y,z], ...]
        directions = [np.array(d, dtype=np.float64) for d in handshake_result.directions]

        # Calculate average direction
        avg_direction = np.mean(directions, axis=0)
        direction_coherence = float(np.linalg.norm(avg_direction))

        # Calculate pairwise similarity (sample first 5)
        pairwise_sims = []
        limit = min(n, 5)
        for i in range(limit):
            for j in range(i + 1, limit):
                sim = abs(np.dot(directions[i], directions[j]))
                pairwise_sims.append(sim)

        avg_pairwise_sim = float(np.mean(pairwise_sims)) if pairwise_sims else 0.0

        return {
            "direction_coherence": direction_coherence,
            "pattern_strength": avg_pairwise_sim,
            "novelty_signal": direction_coherence,
            "noise_signal": (1.0 - direction_coherence) * (1.0 - avg_pairwise_sim),
        }

    def _extract_pyramid_features(self) -> Dict:
        """
        Extract comprehensive features from the pyramid analysis.

        Returns:
            Dictionary with depth, coverage, novelty, coherence, and activation scores
        """
        if not self.levels:
            return {
                "depth": 0,
                "coverage": 0.0,
                "novelty": 1.0,
                "coherence": 0.0,
                "tag_memo_activation": 0.0,
                "expansion_signal": 1.0,
            }

        level0 = self.levels[0]
        handshake = level0.handshake_features

        # Coverage: total explained energy
        coverage = min(1.0, self.total_explained_energy)

        # Coherence: pattern strength from first level
        coherence = handshake["pattern_strength"] if handshake else 0.0

        # Novelty: residual ratio + directional novelty
        residual_ratio = 1.0 - coverage
        directional_novelty = handshake["novelty_signal"] if handshake else 0.0
        novelty = (residual_ratio * 0.7) + (directional_novelty * 0.3)

        return {
            "depth": len(self.levels),
            "coverage": coverage,
            "novelty": novelty,
            "coherence": coherence,
            "tag_memo_activation": coverage * coherence * (1.0 - (handshake["noise_signal"] if handshake else 0.0)),
            "expansion_signal": novelty,
        }

    def _flatten_tag_vectors(self, tags: List[Dict]) -> bytes:
        """Flatten tag vectors into a single bytes buffer for Rust."""
        dim = self._dim
        n = len(tags)
        flattened = np.zeros(n * dim, dtype=np.float32)

        for i, tag in enumerate(tags):
            vec_bytes = tag["vector"]
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            if len(vec) != dim:
                if len(vec) < dim:
                    padded = np.zeros(dim, dtype=np.float32)
                    padded[:len(vec)] = vec
                    vec = padded
                else:
                    vec = vec[:dim]
            start = i * dim
            flattened[start:start + dim] = vec

        return flattened.tobytes()

    def _vector_to_bytes(self, vector: np.ndarray) -> bytes:
        """Convert numpy vector to bytes for Rust."""
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        return vector.tobytes()

    def _magnitude(self, vector: np.ndarray) -> float:
        """Calculate L2 magnitude of a vector."""
        return float(np.linalg.norm(vector))

    def _compile_results(self) -> Dict[str, Any]:
        """Compile analysis results into a structured output."""
        return {
            "levels": [level.to_dict() for level in self.levels],
            "total_explained_energy": self.total_explained_energy,
            "final_residual": self.final_residual.tolist() if self.final_residual is not None else [],
            "features": self.features,
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for invalid input."""
        return {
            "levels": [],
            "total_explained_energy": 0.0,
            "final_residual": [0.0] * self._dim,
            "features": {
                "depth": 0,
                "coverage": 0.0,
                "novelty": 1.0,
                "coherence": 0.0,
                "tag_memo_activation": 0.0,
                "expansion_signal": 1.0,
            },
        }

    def get_residual_at_level(self, level: int) -> Optional[np.ndarray]:
        """
        Get the residual vector at a specific level.

        Note: Full residual vectors are not stored by default to save memory.
        You would need to modify analyze() to store them if needed.
        """
        _ = level  # Reserved for future implementation
        return None

    def get_energy_profile(self) -> List[float]:
        """Get the energy profile across all levels."""
        return [level.residual_energy_ratio for level in self.levels]

    def clear(self) -> None:
        """Clear all stored data."""
        self.levels.clear()
        self.total_explained_energy = 0.0
        self.final_residual = None
        self.features = {}

    def export_state(self) -> Dict:
        """Export pyramid state."""
        return {
            "levels": [level.to_dict() for level in self.levels],
            "total_explained_energy": self.total_explained_energy,
            "final_residual": self.final_residual.tolist() if self.final_residual is not None else [],
            "features": self.features,
            "config": self.config,
        }
