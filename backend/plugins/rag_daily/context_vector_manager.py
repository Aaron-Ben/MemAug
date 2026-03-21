"""
ContextVectorManager - 上下文向量对应映射管理模块

功能：
1. 维护当前会话中所有消息（除最后一条 AI 和用户消息外）的向量映射。
2. 提供模糊匹配技术，处理 AI 或用户对上下文的微小编辑。
3. 为后续的"上下文向量衰减聚合系统"提供底层数据支持。
"""

import hashlib
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np


logger = logging.getLogger(__name__)

class ContextVectorManager:
    """
    管理对话消息的上下文向量映射。

    特性：
    - 模糊匹配技术处理微小编辑
    - 语义向量分段
    - 上下文向量衰减聚合
    - 语义宽度计算
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.85,
        decay_rate: float = 0.75,
        max_context_window: int = 10,
        dimension: int = 1024,
    ):
        """
        初始化上下文向量管理器。

        Args:
            fuzzy_threshold: 模糊匹配阈值 (0.0 ~ 1.0)，用于判断两个文本是否足够相似以复用向量
            decay_rate: 衰减率，用于聚合历史向量
            max_context_window: 限制聚合窗口大小
            dimension: 向量维度
        """
        # 核心映射：normalized_hash -> {vector, role, original_text, timestamp}
        self.vector_map: Dict[str, Dict[str, Any]] = {}

        # 顺序索引：用于按顺序获取向量
        self.history_assistant_vectors: List[np.ndarray] = []
        self.history_user_vectors: List[np.ndarray] = []

        # 配置参数
        self.fuzzy_threshold = fuzzy_threshold
        self.decay_rate = decay_rate
        self.max_context_window = max_context_window
        self.dimension = dimension

    def _generate_hash(self, text: str) -> str:
        """
        生成内容哈希

        Args:
            text: 原始文本

        Returns:
            SHA256 哈希值
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _normalize(self, text: str) -> str:
        """
        标准化文本，用于模糊匹配

        Args:
            text: 原始文本

        Returns:
            标准化后的文本（转小写，去除多余空格）
        """
        # 转小写，去除首尾空格，合并多个空格为单个空格
        normalized = re.sub(r'\s+', ' ', text.strip().lower())
        return normalized

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        简单的字符串相似度算法 (Dice's Coefficient)
        用于处理微小编辑时的模糊匹配

        Args:
            str1: 字符串1
            str2: 字符串2

        Returns:
            相似度 (0.0 ~ 1.0)
        """
        if str1 == str2:
            return 1.0
        if len(str1) < 2 or len(str2) < 2:
            return 0.0

        def get_bigrams(s: str) -> set:
            """获取字符二元组集合"""
            return {s[i:i+2] for i in range(len(s) - 1)}

        b1 = get_bigrams(str1)
        b2 = get_bigrams(str2)

        if not b1 or not b2:
            return 0.0

        intersection = len(b1 & b2)
        return (2.0 * intersection) / (len(b1) + len(b2))

    def _find_fuzzy_match(self, normalized_text: str) -> Optional[np.ndarray]:
        """
        尝试在现有缓存中寻找模糊匹配的向量

        Args:
            normalized_text: 标准化后的文本

        Returns:
            匹配的向量，如果没有找到则返回 None
        """
        for entry in self.vector_map.values():
            similarity = self._calculate_similarity(
                normalized_text,
                self._normalize(entry['original_text'])
            )
            if similarity >= self.fuzzy_threshold:
                return entry['vector']
        return None

    async def update_context(
        self,
        messages: List[Dict],
        embedding_cache: Optional[Dict[str, np.ndarray]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        更新上下文映射

        Args:
            messages: 当前会话的消息数组
            embedding_cache: Embedding 缓存字典 {content: vector}
            options: 选项字典，支持 allow_api
        """
        logger.info(f"[ContextVectorManager] 🔄 Updating context with {len(messages)} messages...")

        if not isinstance(messages, list):
            logger.warning("[ContextVectorManager] ⚠️ Messages is not a list, skipping update")
            return

        options = options or {}
        allow_api = options.get('allow_api', False)                                                                                             
                                                                                                                                                    
        new_assistant_vectors = []
        new_user_vectors = []

        # 识别最后的消息索引以进行排除
        last_user_index = next((
            i for i in range(len(messages) - 1, -1, -1)
            if messages[i].get('role') == 'user'
        ), -1)

        last_ai_index = next((
            i for i in range(len(messages) - 1, -1, -1)
            if messages[i].get('role') == 'assistant'
        ), -1)

        logger.debug(f"[ContextVectorManager] 📊 Last user index: {last_user_index}, Last AI index: {last_ai_index}")

        tasks = []
        for index, msg in enumerate(messages):
            # 排除逻辑：系统消息、最后一个用户消息、最后一个 AI 消息
            role = msg.get('role')
            if role == 'system' or index == last_user_index or index == last_ai_index:
                continue

            # 获取内容
            content = msg.get('content', '')
            if isinstance(content, list):
                # 处理内容是列表的情况（如 multimodal 消息）
                text_items = [item.get('text', '') for item in content if item.get('type') == 'text']
                content = ' '.join(text_items)

            if not content or len(content) < 2:
                continue

            normalized = self._normalize(content)
            content_hash = self._generate_hash(normalized)

            vector = None
            match_source = None

            # 1. 精确匹配
            if content_hash in self.vector_map:
                vector = self.vector_map[content_hash]['vector']
                match_source = "exact"
                cache_hit_count += 1
            # 2. 模糊匹配 (处理微小编辑)
            else:
                vector = self._find_fuzzy_match(normalized)
                if vector is not None:
                    match_source = "fuzzy"

                # 3. 尝试从插件的 Embedding 缓存中获取（不触发 API）
                if vector is None and embedding_cache:
                    vector = embedding_cache.get(content)
                    if vector is not None:
                        match_source = "embedding_cache"

                # 存入映射
                if vector is not None:
                    self.vector_map[content_hash] = {
                        'vector': vector,
                        'role': role,
                        'original_text': content,
                        'timestamp': datetime.now().timestamp()
                    }

            if vector is not None:
                logger.debug(f"[ContextVectorManager] ✨ Msg[{index}] ({role}): matched via {match_source or 'new'}")
                if role == 'assistant':
                    new_assistant_vectors.append(vector)
                elif role == 'user':
                    new_user_vectors.append(vector)

        # 更新历史向量列表
        self.history_assistant_vectors = new_assistant_vectors
        self.history_user_vectors = new_user_vectors

    def aggregate_context(self, role: str = 'assistant') -> Optional[np.ndarray]:
        """                                                                  
        聚合上下文向量（指数衰减加权平均）。                                                                                                      
                                                                                                                                                    
        Args:                                                                                                                                     
            role: 'assistant' 或 'user'                                                                                                           
                                                                                                                                                    
        Returns:
            聚合后的向量，或 None（如果无向量）
        """                                                                                                                                       
        vectors = (self.history_assistant_vectors
                    if role == 'assistant'                                                                                                         
                    else self.history_user_vectors)                                                                                                
    
        if len(vectors) == 0:                                                                                                                     
            return None

        # 限制窗口：只取最近的 max_context_window 个
        if len(vectors) > self.max_context_window:
            vectors = vectors[-self.max_context_window:]
                                                                                                                                                    
        dim = len(vectors[0])
        aggregated = np.zeros(dim, dtype=np.float32)                                                                                              
        total_weight = 0.0
                                                                                                                                                    
        # 从旧到新遍历（idx=0 为最旧）                                                                                                            
        for idx, vector in enumerate(vectors):                                                                                                    
            age = len(vectors) - idx  # 新向量 age 小，权重高                                                                                     
            weight = self.decay_rate ** age                                                                                                       
                                                                                                                                                    
            aggregated += vector * weight                                                                                                         
            total_weight += weight
                                                                                                                                                    
        if total_weight > 0:
            aggregated /= total_weight                                                                                                            
                                                                                                                                                    
        return aggregated

    def compute_logic_depth(vector: np.ndarray, topK: int = 64) -> float:                                                                         
        """                                                                                                                                       
        计算向量逻辑深度（稀疏度/集中度）。                                                                                                       
                                                                                                                                                    
        Args:                                                                                                                                     
            vector: 输入向量 (numpy array)                                                                                                        
            topK: Top-K 维度数量                                                                                                                  
                                                                                                                                                
        Returns:                                                                                                                                  
            逻辑深度值 (0.0 ~ 1.0)                                                                                                                
        """                                                                                                                                       
        if vector is None or len(vector) == 0:                                                                                                    
            return 0.0                                                                                                                            
                                                                                                                                                    
        dim = len(vector)                                                                                                                         
        energies = vector ** 2
        total_energy = np.sum(energies)                                                                                                           
    
        if total_energy < 1e-9:                                                                                                                   
            return 0.0

        sorted_energies = np.sort(energies)[::-1]  # 降序排序                                                                                     
        actual_topK = min(topK, dim)
        topK_energy = np.sum(sorted_energies[:actual_topK])                                                                                       
                                                                                                                                                    
        concentration = topK_energy / total_energy                                                                                                
        expected_uniform = actual_topK / dim                                                                                                      
        L = (concentration - expected_uniform) / (1 - expected_uniform)                                                                           
                
        return max(0.0, min(1.0, L))

    def compute_semantic_width(self, vector: Optional[np.ndarray]) -> float:
        """
        计算语义宽度指数 S
        核心思想：向量的模长反映了语义的确定性/强度

        Args:
            vector: 输入向量

        Returns:
            语义宽度值
        """
        if vector is None:
            return 0.0
        magnitude = np.linalg.norm(vector)
        spread_factor = 1.2  # 可调参数
        return magnitude * spread_factor

    def get_context_summary(self) -> Dict:
        """
        获取当前上下文状态的摘要

        Returns:
            包含上下文统计信息的字典
        """
        return {
            "total_vectors": len(self.vector_map),
            "history_assistant_count": len(self.history_assistant_vectors),
            "history_user_count": len(self.history_user_vectors),
            "fuzzy_threshold": self.fuzzy_threshold,
            "decay_rate": self.decay_rate,
            "max_context_window": self.max_context_window,
        }


