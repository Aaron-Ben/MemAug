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


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """计算余弦相似度"""
    if vec_a.shape != vec_b.shape:
        return 0.0
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        计算余弦相似度

        Args:
            vec_a: 向量A
            vec_b: 向量B

        Returns:
            余弦相似度 (0.0 ~ 1.0)
        """
        return cosine_similarity(vec_a, vec_b)

    def _finalize_segment(
        self,
        vectors: List[np.ndarray],
        texts: List[str],
        roles: List[str],
        start_index: int,
        end_index: int
    ) -> Dict[str, Any]:
        """
        完成分段的计算（计算平均向量并归一化）

        Args:
            vectors: 分段中的向量列表
            texts: 分段中的文本列表
            roles: 分段中的角色列表
            start_index: 起始索引
            end_index: 结束索引

        Returns:
            分段字典 {vector, text, roles, range, count}
        """
        if not vectors:
            return None

        # 计算平均向量
        count = len(vectors)
        avg_vec = np.mean(vectors, axis=0)

        # 归一化
        norm = np.linalg.norm(avg_vec)
        if norm > 1e-9:
            avg_vec = avg_vec / norm

        return {
            'vector': avg_vec,
            'text': '\n'.join(texts),
            'roles': list(set(roles)),  # 去重角色
            'range': [start_index, end_index],
            'count': count
        }

    def update_context(
        self,
        messages: List[Dict],
        embedding_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        更新上下文映射

        Args:
            messages: 当前会话的消息数组
            embedding_cache: Embedding 缓存字典 {content: vector}
            allow_api: 是否允许 API 调用获取新向量（当前实现未使用）
        """
        logger.info(f"[ContextVectorManager] 🔄 Updating context with {len(messages)} messages...")

        if not isinstance(messages, list):
            logger.warning("[ContextVectorManager] ⚠️ Messages is not a list, skipping update")
            return

        new_assistant_vectors = []
        new_user_vectors = []
        fuzzy_match_count = 0
        cache_hit_count = 0

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

        for index, msg in enumerate(messages):
            # 排除逻辑：系统消息、最后一个用户消息、最后一个 AI 消息
            role = msg.get('role')
            if role == 'system':
                continue
            if index == last_user_index or index == last_ai_index:
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
                    fuzzy_match_count += 1

                # 3. 尝试从插件的 Embedding 缓存中获取（不触发 API）
                if vector is None and embedding_cache:
                    vector = embedding_cache.get(content)
                    if vector is not None:
                        match_source = "embedding_cache"

                # 4. 如果缓存也没有，且允许 API，则请求新向量（触发 API）
                # 注意：当前实现中，API 调用应在外部处理

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

        # 输出详细统计
        logger.info(
            f"[ContextVectorManager] ✅ Context updated: "
            f"{len(self.history_assistant_vectors)} AI vectors, "
            f"{len(self.history_user_vectors)} user vectors, "
            f"{len(self.vector_map)} total entries in cache"
        )
        if fuzzy_match_count > 0:
            logger.info(f"[ContextVectorManager] 🎭 Fuzzy matches: {fuzzy_match_count}, Cache hits: {cache_hit_count}")

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

    def segment_context(
        self,
        messages: List[Dict],
        similarity_threshold: float = 0.70
    ) -> List[Dict[str, Any]]:
        """
        基于语义向量的上下文分段 (Semantic Segmentation)
        将连续的、高相似度的消息归并为一个段落 (Segment/Topic)

        Args:
            messages: 消息列表 (通常是 history)
            similarity_threshold: 分段阈值，低于此值则断开 (默认 0.70)

        Returns:
            分段列表，每个分段包含 {vector, text, role, range, count}
        """
        logger.debug(f"[ContextVectorManager] 📊 Segmenting context with threshold={similarity_threshold}")

        # 重新构建有序序列
        sequence = []
        for index, msg in enumerate(messages):
            # 跳过系统消息和无关消息
            if msg.get('role') == 'system':
                continue

            # 获取内容
            content = msg.get('content', '')
            if isinstance(content, list):
                text_items = [item.get('text', '') for item in content if item.get('type') == 'text']
                content = ' '.join(text_items)

            if not content or len(content) < 2:
                continue

            normalized = self._normalize(content)
            content_hash = self._generate_hash(normalized)

            # 尝试精确匹配
            entry = self.vector_map.get(content_hash)

            # 尝试模糊匹配 (如果精确匹配失败)
            if entry is None:
                fuzzy_vector = self._find_fuzzy_match(normalized)
                if fuzzy_vector is not None:
                    entry = {
                        'vector': fuzzy_vector,
                        'role': msg.get('role'),
                        'original_text': content,
                    }

            if entry and entry.get('vector') is not None:
                sequence.append({
                    'index': index,
                    'role': msg.get('role'),
                    'text': content,
                    'vector': entry['vector']
                })

        if not sequence:
            return []

        # 执行分段
        segments = []
        current_segment = {
            'vectors': [sequence[0]['vector']],
            'texts': [sequence[0]['text']],
            'start_index': sequence[0]['index'],
            'end_index': sequence[0]['index'],
            'roles': [sequence[0]['role']]
        }

        for i in range(1, len(sequence)):
            curr = sequence[i]
            prev = sequence[i - 1]

            # 计算与上一条的相似度
            sim = self._cosine_similarity(prev['vector'], curr['vector'])

            # 角色变化也可以作为分段的弱信号，但在这里我们主要看语义
            # 如果相似度高，即使角色不同也可以合并（例如连续的问答对，讨论同一个话题）
            # 如果相似度低，即使角色相同也应该断开

            if sim >= similarity_threshold:
                # 合并
                current_segment['vectors'].append(curr['vector'])
                current_segment['texts'].append(curr['text'])
                current_segment['end_index'] = curr['index']
                current_segment['roles'].append(curr['role'])
            else:
                # 断开，保存旧段
                segments.append(self._finalize_segment(
                    current_segment['vectors'],
                    current_segment['texts'],
                    current_segment['roles'],
                    current_segment['start_index'],
                    current_segment['end_index']
                ))
                # 开启新段
                current_segment = {
                    'vectors': [curr['vector']],
                    'texts': [curr['text']],
                    'start_index': curr['index'],
                    'end_index': curr['index'],
                    'roles': [curr['role']]
                }

        # 保存最后一个段
        segments.append(self._finalize_segment(
            current_segment['vectors'],
            current_segment['texts'],
            current_segment['roles'],
            current_segment['start_index'],
            current_segment['end_index']
        ))

        logger.info(f"[ContextVectorManager] ✅ Segmentation complete: {len(segments)} segments created")
        for i, seg in enumerate(segments[:10]):
            text_preview = seg['text'][:60].replace('\n', ' ')
            logger.info(f"[ContextVectorManager] 📝 Segment[{i}]: {seg['count']} msgs, roles={seg['roles']}, text=\"{text_preview}...\"")
        if len(segments) > 10:
            logger.debug(f"[ContextVectorManager]   ... and {len(segments) - 10} more segments")

        return segments

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


