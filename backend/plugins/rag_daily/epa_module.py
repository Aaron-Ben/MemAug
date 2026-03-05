"""
Embedding Projection Analysis (EPA) Module for RAG Daily Plugin.

Implements weighted PCA projection and K-Means clustering for semantic space analysis.
Provides cross-domain resonance detection and semantic clustering capabilities.

Physics-Optimized Edition:
- Weighted centered PCA using high-performance Rust SVD
- Robust K-Means clustering
- Cross-domain resonance detection based on energy co-occurrence
"""

import logging
import json
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import numpy as np


logger = logging.getLogger(__name__)


def _float_array_to_bytes(array: np.ndarray) -> bytes:
    """Convert numpy float array to bytes for Rust interop."""
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return array.tobytes()


class EPAModule:
    """
    Embedding Projection Analysis module.

    Features:
    - Weighted PCA for semantic space projection (using Rust SVD)
    - K-Means clustering for semantic groups
    - Cross-domain resonance detection
    - Orthogonal basis vector management
    """

    def __init__(
        self, db, config: Optional[Dict[str, Any]] = None):
        """
        初始化 EPA 类

        Args:
            db: 数据库连接/对象
            config: 配置字典，包含以下可选参数：
                - maxBasisDim / max_basis_dim: 最大基维度 (默认 64)
                - minVarianceRatio / min_variance_ratio: 最小方差比 (默认 0.01)
                - clusterCount / cluster_count: 聚类数量 (默认 32)
                - dimension: 向量维度 (默认 1024)
                - strictOrthogonalization / strict_orthogonalization: 严格正交化 (默认 True)
                - vexusIndex / vexus_index: 索引 (默认 None)
        """
        # 初始化数据库连接
        self.db = db

        # 默认配置
        default_config = {
            'max_basis_dim': 64,
            'min_variance_ratio': 0.01,
            'cluster_count': 32,
            'dimension': 1024,
            'strict_orthogonalization': True,
            'vexus_index': None,
        }

        # 合并用户配置
        if config is not None:
            default_config.update(config)

        # 最终配置赋值
        self.config = default_config

        # Orthogonal basis vectors (U matrix from SVD)
        self.ortho_basis: Optional[List[np.ndarray]] = None

        # Mean vector for centering
        self.basis_mean: Optional[np.ndarray] = None

        # Basis energies (singular values)
        self.basis_energies: Optional[np.ndarray] = None

        # Basis labels
        self.basis_labels: Optional[List[str]] = None

        # Initialization flag
        self.initialized: bool = False


    async def initialize(self) -> bool:
        """
        Initialize EPA basis vectors from tags or cached storage.

        Args:
            tags_vectors: Optional list of tag vectors to compute basis from
            tag_names: Optional list of tag names corresponding to vectors
            kv_store_get_func: Optional function to retrieve cached basis from KV store
            kv_store_set_func: Optional function to save computed basis to KV store
        """
        logger.info("[EPA] 🧠 Initializing orthogonal basis (Weighted PCA)...")

        # Try to load from cache first

        try:
            # 尝试从缓存加载
            if await self._load_from_cache():
                logger.info('[EPA] 💾 Loaded basis from cache.')
                self.initialized = True
                return True

            # 从数据库获取标签数据
            # 注意：这里需要根据你的数据库类型调整查询方式
            # 如果使用 sqlite3，需要同步执行，或者使用 aiosqlite
            tags = self.db.execute(
                "SELECT id, name, vector FROM tags WHERE vector IS NOT NULL"
            ).fetchall()
            
            # 转换标签数据格式（假设 vector 是二进制或序列化的数组）
            # 你可能需要根据实际存储格式调整这里的解析逻辑
            processed_tags = []
            for tag in tags:
                tag_id, name, vector = tag
                # 解析向量（示例：如果是 bytes，转换为 numpy 数组）
                if isinstance(vector, bytes):
                    vec = np.frombuffer(vector, dtype=np.float32)
                elif isinstance(vector, list):
                    vec = np.array(vector, dtype=np.float32)
                else:
                    vec = vector
                processed_tags.append({
                    'id': tag_id,
                    'name': name,
                    'vector': vec
                })
            
            if len(processed_tags) < 8:
                logger.warning('[EPA] ⚠️ Not enough tags (need at least 8)')
                return False

            # 1. 鲁棒 K-Means 聚类 (提取加权质心)
            cluster_count = min(len(processed_tags), self.config['cluster_count'])
            cluster_data = self._cluster_tags(processed_tags, cluster_count)

            # 2. 计算 SVD (加权中心化 PCA)
            svd_result = self._compute_weighted_pca(cluster_data)

            U = svd_result['U']
            S = svd_result['S']
            mean_vector = svd_result['mean']
            labels = svd_result.get('labels', None)
            
            # 3. 选择主成分
            K = self._select_basis_dimension(S)
            
            # 保存结果
            self.ortho_basis = U[:K]
            self.basis_energies = S[:K]
            self.basis_mean = mean_vector
            self.basis_labels = labels[:K] if labels is not None else cluster_data['labels'][:K]
            
            # 保存到缓存
            await self.save_to_cache()
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f'[EPA] ❌ Init failed: {str(e)}', exc_info=True)
            return False


    def _cluster_tags(
        self,
        tags: List[Dict],
        k: int,
    ) -> Dict:
        """
        Perform robust K-Means clustering on tag vectors.

        Args:
            tags: List of tag dictionaries with 'vector' and 'name' keys
            k: Number of clusters

        Returns:
            Dictionary with vectors, labels, and weights
        """
        if not tags or len(tags) < k:
            k = max(1, len(tags))

        tags_array = np.array([t['vector'] for t in tags], dtype=np.float32)
        tag_names = [t.get('name', 'unknown') for t in tags]
        n = len(tags_array)

        # Initialize centroids using Forgy method (random selection)
        indices = np.random.choice(n, k, replace=False)
        centroids = tags_array[indices].copy()

        # Normalize centroids
        for i in range(k):
            norm = np.linalg.norm(centroids[i])
            if norm > 1e-9:
                centroids[i] /= norm

        cluster_sizes = np.zeros(k, dtype=np.float32)
        max_iter = 50
        tolerance = 1e-4

        for iteration in range(max_iter):
            clusters = [[] for _ in range(k)]
            movement = 0.0

            # Assign vectors to nearest centroid
            for vec in tags_array:
                # Calculate similarities (dot product for normalized vectors)
                similarities = np.dot(centroids, vec)
                best_k = np.argmax(similarities)
                clusters[best_k].append(vec)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                if len(clusters[i]) > 0:
                    cluster_array = np.array(clusters[i], dtype=np.float32)
                    new_centroids[i] = np.mean(cluster_array, axis=0)

                    # Normalize
                    norm = np.linalg.norm(new_centroids[i])
                    if norm > 1e-9:
                        new_centroids[i] /= norm

                    # Calculate movement
                    dist_sq = np.sum((new_centroids[i] - centroids[i]) ** 2)
                    movement += dist_sq
                else:
                    new_centroids[i] = centroids[i]

                cluster_sizes[i] = len(clusters[i])

            centroids = new_centroids

            if movement < tolerance:
                logger.debug(f"[EPA] K-Means converged at iter {iteration}")
                break

        # Name clusters by finding closest tag
        labels = []
        for centroid in centroids:
            best_sim = -float('inf')
            closest_name = 'Unknown'
            for i, tag in enumerate(tags_array):
                sim = np.dot(centroid, tag)
                if sim > best_sim:
                    best_sim = sim
                    closest_name = tag_names[i]
            labels.append(closest_name)

        self.cluster_centers = centroids

        return {
            "vectors": centroids,
            "labels": labels,
            "weights": cluster_sizes,
        }

    def _compute_weighted_pca(self, cluster_data: Dict) -> Dict:
        """
        Compute weighted PCA using Power Iteration on Gram matrix.

        This implementation follows the JavaScript version:
        1. Build Gram matrix (n x n) instead of full covariance matrix (dim x dim)
        2. Use Power Iteration with Re-orthogonalization to find eigenvectors
        3. Use Deflation to remove found components
        4. Map Gram eigenvectors back to original space

        Args:
            cluster_data: Dictionary with vectors, labels, and weights

        Returns:
            Dictionary with U, S, mean, and labels
        """
        vectors = cluster_data["vectors"]
        weights = cluster_data["weights"]
        labels = cluster_data["labels"]

        n = len(vectors)
        dim = self.config['dimension']

        # Convert to float32 array
        vectors_array = np.array(vectors, dtype=np.float32)
        weights_array = np.array(weights, dtype=np.float32)

        # Step 1: Compute weighted mean
        total_weight = np.sum(weights_array)
        mean_vector = np.zeros(dim, dtype=np.float32)
        for i in range(n):
            mean_vector += vectors_array[i] * weights_array[i]
        mean_vector /= total_weight

        # Step 2: Center the data with weights (build X_centered_scaled)
        # X_centered_scaled[i] = sqrt(w_i) * (v_i - mean)
        centered_scaled = np.zeros_like(vectors_array)
        for i in range(n):
            scale = np.sqrt(weights_array[i])
            centered_scaled[i] = (vectors_array[i] - mean_vector) * scale

        # Step 3: Build Gram Matrix (n x n)
        # G = X_centered_scaled @ X_centered_scaled.T
        gram = centered_scaled @ centered_scaled.T

        # Step 4: Power Iteration with Re-orthogonalization and Deflation
        gram_eigenvectors = []  # Eigenvectors of Gram matrix (n-dimensional)
        eigenvalues = []         # Corresponding eigenvalues

        max_basis = min(n, self.config['max_basis_dim'], dim)
        gram_copy = gram.copy().astype(np.float64)

        for k in range(max_basis):
            # Use power iteration to find dominant eigenvector
            result = self._power_iteration(
                matrix=gram_copy,
                n=n,
                existing_basis=gram_eigenvectors,
                max_iter=100,
                tol=1e-6,
                strict_orthogonalization=True
            )

            v = result['vector']   # n-dimensional eigenvector of Gram matrix
            value = result['value'] # Corresponding eigenvalue

            # Stop if eigenvalue is too small
            if value < 1e-6:
                logger.debug(f"[EPA] Eigenvalue {k} = {value:.2e} < 1e-6, stopping")
                break

            gram_eigenvectors.append(v)
            eigenvalues.append(value)

            # Deflation: G_new = G_old - lambda * v * v^T
            # Remove the found component from the Gram matrix
            gram_copy = gram_copy - value * np.outer(v, v)

        # Step 5: Map Gram eigenvectors back to original space (dim-dimensional)
        # U_pca[i] = X_centered_scaled.T @ v[i] / sqrt(lambda[i])
        U = []
        for i, v in enumerate(gram_eigenvectors):
            lambda_i = max(eigenvalues[i], 1e-12)  # Avoid division by zero

            # Linear combination: U_pca = X^T * v / sqrt(lambda)
            # basis[j] = sum_i(X[i][j] * v[i]) for all i
            basis = np.zeros(dim, dtype=np.float32)

            for vec_idx in range(n):
                weight = v[vec_idx]
                if abs(weight) > 1e-9:
                    basis += weight * centered_scaled[vec_idx]

            # Normalize
            basis /= np.sqrt(lambda_i)

            # Re-normalize to unit length
            norm = np.linalg.norm(basis)
            if norm > 1e-9:
                basis /= norm

            U.append(basis)

        logger.debug(f"[EPA] Power Iteration PCA: found {len(U)} basis vectors, eigenvalues: {eigenvalues[:5]}...")

        return {
            "U": U,
            "S": np.array(eigenvalues, dtype=np.float32),
            "mean": mean_vector,
            "labels": labels,
        }

    def _select_basis_dimension(self, S: np.ndarray) -> int:
        """
        Select basis dimension based on explained variance ratio.

        Args:
            S: Singular values (eigenvalues)

        Returns:
            Selected dimension K
        """
        if len(S) == 0 or (total := S.sum()) == 0:
            return 8

        explained = np.cumsum(S) / total
        k = np.argmax(explained > 0.95) + 1

        return max(k, 8)

    def _power_iteration(
        self,
        matrix: np.ndarray,
        n: int,
        existing_basis: Optional[List[np.ndarray]] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        strict_orthogonalization: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Power iteration method to compute the dominant eigenvalue and eigenvector.

        This is a pure Python implementation of the power iteration algorithm,
        which finds the largest eigenvalue in magnitude and its corresponding
        eigenvector for a given square matrix.

        Args:
            matrix: Square matrix (n x n) as flattened numpy array or 2D array
            n: Dimension of the matrix
            existing_basis: List of existing eigenvectors for orthogonalization (deflation)
            max_iter: Maximum number of iterations (default: 100)
            tol: Convergence tolerance (default: 1e-6)
            strict_orthogonalization: Whether to enforce strict orthogonalization (default: True)

        Returns:
            Dictionary with:
                - 'vector': Eigenvector (numpy array)
                - 'value': Eigenvalue (float)

        Algorithm:
            1. Initialize random vector
            2. Iterate: v = M * v / ||M * v||
            3. Compute Rayleigh quotient as eigenvalue estimate
            4. (Optional) Re-orthogonalize against existing basis

        Reference:
            Corresponds to JavaScript _powerIteration method.
        """
        # Ensure matrix is 2D
        if matrix.ndim == 1:
            matrix = matrix.reshape(n, n)

        # Random initialization
        v = np.random.randn(n).astype(np.float64)
        v /= (np.linalg.norm(v) + 1e-12)

        last_val = 0.0

        for iteration in range(max_iter):
            # Matrix-vector multiplication: w = M * v
            w = matrix @ v

            # 🔥 关键优化：Re-orthogonalization (Gram-Schmidt against existing)
            # 在归一化之前进行正交化，防止幂迭代收敛到已经找到的主成分上
            # 解决 Deflation 精度丢失问题
            if strict_orthogonalization and existing_basis:
                for prev_v in existing_basis:
                    # Subtract projection onto existing eigenvector
                    dot = np.dot(w, prev_v)
                    w = w - dot * prev_v

            # Normalize
            mag = np.linalg.norm(w)
            if mag < 1e-12:
                # Converged to zero vector (e.g., matrix is singular)
                break

            v = w / mag

            # Rayleigh quotient: eigenvalue estimate (after normalization)
            # λ = (v^T * M * v) / (v^T * v) = v^T * w_original (but we use v @ M @ v)
            val = np.dot(v, matrix @ v)

            # Check convergence
            if abs(val - last_val) < tol:
                last_val = val
                break

            last_val = val

        return {
            "vector": v.astype(np.float32),
            "value": float(last_val),
        }

    def project(self, vector: np.ndarray) -> Dict[str, Any]:
        """
        Project a vector onto the semantic space defined by basis vectors.

        Uses Rust high-performance projection when available.

        Args:
            vector: Input vector to project

        Returns:
            Dictionary with projections, probabilities, entropy, dominant_axes
        """
        if not self.initialized or self.ortho_basis is None:
            return self._empty_result()

        vec = vector.astype(np.float32) if vector.dtype != np.float32 else vector
        dim = len(vec)
        K = len(self.ortho_basis)

        if dim != self.config['dimension']:
            raise ValueError(f"Vector dimension mismatch: expected {self.config['dimension']}, got {dim}")

        projections = None
        probabilities = None
        entropy = None

        # Use Rust high-performance projection
        if self.config['vexus_index'] is not None:
            try:
                # Flatten basis vectors
                flattened_basis = np.zeros(K * dim, dtype=np.float32)
                for i, basis in enumerate(self.ortho_basis):
                    flattened_basis[i * dim:(i + 1) * dim] = basis

                result = self.config['vexus_index'].project(
                    vector=_float_array_to_bytes(vec),
                    flattened_basis=_float_array_to_bytes(flattened_basis),
                    mean_vector=_float_array_to_bytes(self.basis_mean),
                    k=K,
                )

                projections = np.array(result.projections, dtype=np.float32)
                probabilities = np.array(result.probabilities, dtype=np.float32)
                entropy = result.entropy
                # total_energy is available in result but not currently used

            except Exception as e:
                logger.warning(f"[EPA] Rust projection failed: {e}")
                raise

        if projections is None:
            return self._empty_result()

        # Normalize entropy
        normalized_entropy = entropy / np.log2(K) if K > 1 else 0

        # Extract dominant axes
        dominant_axes = []
        for k in range(K):
            if probabilities[k] > 0.05:  # Threshold lowered for centered data
                dominant_axes.append({
                    "index": k,
                    "label": self.basis_labels[k] if self.basis_labels else f"axis_{k}",
                    "energy": float(probabilities[k]),
                    "projection": float(projections[k]),
                })

        dominant_axes.sort(key=lambda x: x["energy"], reverse=True)

        return {
            "projections": projections,
            "probabilities": probabilities,
            "entropy": float(normalized_entropy),
            "logic_depth": 1.0 - float(normalized_entropy),
            "dominant_axes": dominant_axes,
        }

    def detect_cross_domain_resonance(self, vector: np.ndarray) -> Dict[str, Any]:
        """
        Detect cross-domain resonance by checking co-activation on orthogonal axes.

        Logic: Detect if multiple orthogonal semantic axes are strongly activated
        simultaneously. This indicates cross-domain semantic resonance.

        Args:
            vector: Input vector

        Returns:
            Dictionary with resonance score and bridges
        """
        projection_result = self.project(vector)
        dominant_axes = projection_result.get("dominant_axes", [])

        if len(dominant_axes) < 2:
            return {"resonance": 0.0, "bridges": []}

        bridges = []
        top_axis = dominant_axes[0]

        # Check co-activation with other axes
        for i in range(1, len(dominant_axes)):
            secondary_axis = dominant_axes[i]

            # Geometric mean energy: sqrt(E1 * E2)
            # Represents simultaneous activation strength
            co_activation = np.sqrt(top_axis["energy"] * secondary_axis["energy"])

            if co_activation > 0.15:  # Threshold for resonance
                bridges.append({
                    "from": top_axis["label"],
                    "to": secondary_axis["label"],
                    "strength": float(co_activation),
                    "balance": min(top_axis["energy"], secondary_axis["energy"]) /
                             max(top_axis["energy"], secondary_axis["energy"]),
                })

        total_resonance = sum(b["strength"] for b in bridges)

        return {
            "resonance": float(total_resonance),
            "bridges": bridges,
        }

    def export_state(self) -> Dict:
        """
        Export EPA state for caching.

        Returns:
            Dictionary with EPA state
        """
        state = {
            "config": {
                "max_basis_dim": self.config['max_basis_dim'],
                "cluster_count": self.config['cluster_count'],
                "dimension": self.config['dimension'],
            }
        }

        if self.ortho_basis is not None:
            state["basis"] = [
                base64.b64encode(vec.tobytes()).decode('utf-8')
                for vec in self.ortho_basis
            ]

        if self.basis_mean is not None:
            state["mean"] = base64.b64encode(self.basis_mean.tobytes()).decode('utf-8')

        if self.basis_energies is not None:
            state["energies"] = self.basis_energies.tolist()

        if self.basis_labels is not None:
            state["labels"] = self.basis_labels

        return state

    async def _save_to_cache(self) -> None:
        """
        将正交基、均值向量等数据序列化后保存到数据库缓存
        核心逻辑：
        1. Float32数组 → bytes → Base64字符串
        2. 其他数据直接JSON序列化
        3. 插入/替换数据库 kv_store 表
        """
        try:
            # 1. 序列化正交基（二维数组：K x dimension）
            # 先将每个基向量转为 bytes，再编码为 Base64 字符串
            basis_b64 = []
            for basis_vec in self.ortho_basis:
                # 确保是 float32 类型，转为 bytes
                vec_bytes = basis_vec.astype(np.float32).tobytes()
                basis_b64.append(base64.b64encode(vec_bytes).decode('utf-8'))
            
            # 2. 序列化均值向量
            mean_bytes = self.basis_mean.astype(np.float32).tobytes()
            mean_b64 = base64.b64encode(mean_bytes).decode('utf-8')
            
            # 3. 整理缓存数据（兼容JSON序列化）
            cache_data = {
                'basis': basis_b64,
                'mean': mean_b64,
                'energies': self.basis_energies.tolist(),  # 数组转列表
                'labels': self.basis_labels,
                'timestamp': int(datetime.now().timestamp() * 1000),  # 毫秒级时间戳
                'tagCount': self._get_tag_count()  # 单独封装获取标签数量
            }
            
            # 4. 写入数据库（INSERT OR REPLACE）
            # 注意：如果使用同步数据库（如 sqlite3），异步方法中需用线程池执行
            await self._execute_db(
                "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                ('epa_basis_cache', json.dumps(cache_data))
            )
            
            logger.info('[EPA] ✅ Cache saved successfully')
            
        except Exception as e:
            logger.error(f'[EPA] ❌ Save cache error: {str(e)}', exc_info=True)

    async def _load_from_cache(self) -> bool:
        """
        从数据库缓存加载并反序列化正交基数据
        Returns:
            bool: 加载成功返回True，失败返回False
        """
        try:
            # 1. 从数据库读取缓存数据
            row = await self._execute_db(
                "SELECT value FROM kv_store WHERE key = ?",
                ('epa_basis_cache',),
                fetch_one=True
            )
            if not row:
                logger.info('[EPA] 📭 No cache found in database')
                return False
            
            # 2. 解析JSON
            cache_data = json.loads(row[0])  # row是元组，value在第一个位置
            
            # 3. 校验缓存格式（兼容旧版本）
            if 'mean' not in cache_data:
                logger.warning('[EPA] ⚠️ Old cache format (missing mean), skip loading')
                return False
            
            # 4. 反序列化正交基
            ortho_basis = []
            for b64_str in cache_data['basis']:
                # Base64 → bytes → float32数组
                vec_bytes = base64.b64decode(b64_str)
                vec = np.frombuffer(vec_bytes, dtype=np.float32)
                ortho_basis.append(vec)
            self.ortho_basis = np.array(ortho_basis)  # 转为二维NumPy数组
            
            # 5. 反序列化均值向量
            mean_bytes = base64.b64decode(cache_data['mean'])
            self.basis_mean = np.frombuffer(mean_bytes, dtype=np.float32)
            
            # 6. 反序列化能量值和标签
            self.basis_energies = np.array(cache_data['energies'], dtype=np.float32)
            self.basis_labels = cache_data['labels']
            
            logger.info('[EPA] ✅ Cache loaded successfully')
            return True
            
        except Exception as e:
            logger.error(f'[EPA] ❌ Load cache error: {str(e)}', exc_info=True)
            return False

    # ------------------- 辅助方法（适配不同数据库） -------------------
    def _get_tag_count(self) -> int:
        """获取标签总数（同步方法）"""
        # 同步执行数据库查询（根据实际数据库类型调整）
        cursor = self.db.execute("SELECT COUNT(*) as count FROM tags")
        return cursor.fetchone()[0]

    async def _execute_db(self, sql: str, params: tuple = (), fetch_one: bool = False):
        """
        异步执行数据库操作（适配同步数据库驱动）
        Args:
            sql: SQL语句
            params: SQL参数
            fetch_one: 是否只获取一条结果
        Returns:
            查询结果（fetch_one=True返回单条，否则返回全部）
        """
        # 如果使用同步数据库（如 sqlite3），用线程池异步执行
        loop = asyncio.get_running_loop()
        cursor = await loop.run_in_executor(
            None,
            lambda: self.db.execute(sql, params)
        )
        self.db.commit()  # 提交事务（写操作需要）
        
        if fetch_one:
            return cursor.fetchone()
        return cursor.fetchall()

    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            "projections": None,
            "probabilities": None,
            "entropy": 1.0,
            "logic_depth": 0.0,
            "dominant_axes": [],
        }
