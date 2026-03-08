"""
RAGDailyPlugin - 日记检索插件

根据时间表达式从日记中检索相关内容，作为用户主入口文件。
用户可根据需要扩展此文件。

Features:
- 时间表达式解析
- 上下文向量管理
- 嵌入投影分析 (EPA)
- 残差金字塔分析
- 智能结果去重
"""

import asyncio
import copy
import json
import math
import os
import re
import sys
import aiofiles
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, TYPE_CHECKING
from datetime import datetime
import logging

# 动态导入（支持直接加载和包导入两种方式）
try:
    from .time_parser import TimeExpressionParser, TimeRange
    from .context_vector_manager import ContextVectorManager
except ImportError:
    # 直接加载时使用绝对路径导入
    plugin_dir = Path(__file__).parent
    sys.path.insert(0, str(plugin_dir))
    from time_parser import TimeExpressionParser, TimeRange
    from context_vector_manager import ContextVectorManager

from app.services.embedding import EmbeddingService

# 类型提示导入（避免运行时循环导入）
if TYPE_CHECKING:
    from app.vector_index import VectorIndex

logger = logging.getLogger(__name__)


DEFAULT_TIMEZONE = "Asia/Shanghai"
dailyNoteRootPath = Path(__file__).parent / "daily_notes"


# ==================== 辅助函数 ====================

def _get_attr(obj, key, default=''):
    """安全获取对象属性（支持 dataclass 和 dict）"""
    if hasattr(obj, key):
        value = getattr(obj, key)
        return value if value is not None else default
    elif isinstance(obj, dict):
        return obj.get(key, default)
    return default


class RAGDiaryPlugin:
    def __init__(self):
        self.name = 'RAGDiaryPlugin'
        self.vector_db_manager: Optional['VectorIndex'] = None
        self.rag_config: Dict[str, Any] = {}
        self.rerank_config: Dict[str, Any] = {}
        self.push_vcp_info: Optional[callable] = None
        self.time_parser = TimeExpressionParser('zh-CN', DEFAULT_TIMEZONE)
        # 修复: ContextVectorManager 不需要插件实例参数
        self.context_vector_manager = ContextVectorManager()
        self.is_initialized = False
        self.context_vector_allow_api: bool = False

    async def load_config(self):
        """加载插件配置（.env 文件和 rag_tags.json）"""
        # --- 加载插件独立的 .env 文件 ---
        env_path = Path(__file__).parent / "config.env"
        
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)

        # 解析上下文向量API开关
        context_vector_allow_api = os.getenv("CONTEXT_VECTOR_ALLOW_API_HISTORY", "false").lower()
        self.context_vector_allow_api = context_vector_allow_api == "true"

        # --- 加载 Rerank 配置 ---
        self.rerank_config = {
            "url": os.getenv("RerankUrl", ""),
            "api_key": os.getenv("RerankApi", ""),
            "model": os.getenv("RerankModel", ""),
            "multiplier": float(os.getenv("RerankMultiplier", 2.0)),
            "max_tokens": int(os.getenv("RerankMaxTokensPerBatch", 30000))  # 蛇形：maxTokens → max_tokens
        }
        
        # 移除启动时检查，改为在调用时实时检查
        if self.rerank_config["url"] and self.rerank_config["api_key"] and self.rerank_config["model"]:
            print('[RAGDiaryPlugin] Rerank feature is configured.')

        config_path = Path(__file__).parent / "rag_tags.json"

        try:
            try:
                # Python 异步读取文件（需使用 aiofiles 库：pip install aiofiles）
                import aiofiles
                async with aiofiles.open(config_path, 'r', encoding='utf-8') as f:
                    config_data = await f.read()
                self.rag_config = json.loads(config_data)
            except FileNotFoundError:
                print('[RAGDiaryPlugin] 缓存文件不存在或已损坏，将重新构建。')
            except json.JSONDecodeError:
                print('[RAGDiaryPlugin] 缓存文件不存在或已损坏，将重新构建。')
        except Exception as error:
            print(f'[RAGDiaryPlugin] 加载配置文件或处理缓存时发生严重错误: {error}')
            self.rag_config = {}

    async def initialize(self, config: Dict[str, Any], dependencies: Dict[str, Any]):
        """初始化插件，注入依赖并加载配置"""
        if "vectorDBManager" in dependencies:
            self.vector_db_manager = dependencies["vectorDBManager"]
            print('[RAGDiaryPlugin] vector_db_manager 依赖已注入。')
        
        vcp_log_functions = dependencies.get("vcpLogFunctions")
        if vcp_log_functions and callable(vcp_log_functions.get("pushVcpInfo")):
            self.push_vcp_info = vcp_log_functions["pushVcpInfo"]
            print('[RAGDiaryPlugin] push_vcp_info 依赖已成功注入。')
        else:
            print('[RAGDiaryPlugin] 警告：push_vcp_info 依赖注入失败或未提供。')

        print('[RAGDiaryPlugin] 开始加载配置...')
        await self.load_config()

        self.is_initialized = True

    
    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        :param vec_a: 向量A
        :param vec_b: 向量B
        :return: 余弦相似度（0~1），无效输入返回0
        """
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        
        dot_product = 0.0
        norm_a = 0.0
        norm_b = 0.0
        
        for a, b in zip(vec_a, vec_b):
            dot_product += a * b
            norm_a += a * a
            norm_b += b * b
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))

    def _get_weighted_average_vector(
        self, 
        vectors: List[List[float]], 
        weights: List[float]
    ) -> Optional[List[float]]:
        """
        计算多个向量的加权平均向量
        :param vectors: 向量列表
        :param weights: 对应权重列表
        :return: 加权平均向量，无有效向量返回None
        """
        # 1. 过滤掉无效的向量及其对应的权重
        valid_vectors = []
        valid_weights = []
        
        for vec, w in zip(vectors, weights):
            if vec and len(vec) > 0:
                valid_vectors.append(vec)
                valid_weights.append(w if w is not None else 0.0)
        
        if not valid_vectors:
            return None
        if len(valid_vectors) == 1:
            return valid_vectors[0]
        
        # 2. 归一化权重
        weight_sum = sum(valid_weights)
        if weight_sum == 0:
            print('[RAGDiaryPlugin] Weight sum is zero, using equal weights.')
            equal_weight = 1.0 / len(valid_vectors)
            valid_weights = [equal_weight] * len(valid_vectors)
            weight_sum = 1.0
        
        normalized_weights = [w / weight_sum for w in valid_weights]
        dimension = len(valid_vectors[0])
        result = [0.0] * dimension
        
        # 3. 计算加权平均值
        for vec, weight in zip(valid_vectors, normalized_weights):
            if len(vec) != dimension:
                print('[RAGDiaryPlugin] Vector dimensions do not match. Skipping mismatched vector.')
                continue
            for j in range(dimension):
                result[j] += vec[j] * weight
        
        return result


    async def get_diary_content(self, character_name: str) -> str:
        """
        异步读取指定角色的日记本内容（整合所有 .txt/.md 文件）

        优化：使用并行读取提高性能

        :param character_name: 角色名
        :return: 整合后的日记内容（含错误提示）
        """
        character_dir_path = dailyNoteRootPath / character_name
        character_diary_content = f"[{character_name}日记本内容为空]"

        try:
            # 读取目录下的文件列表
            files = []
            if os.path.exists(character_dir_path) and os.path.isdir(character_dir_path):
                files = [f for f in os.listdir(character_dir_path) if os.path.isfile(character_dir_path / f)]

            # 过滤并排序相关文件（.txt/.md，不区分大小写）
            relevant_files = [
                file for file in files
                if file.lower().endswith(('.txt', '.md'))
            ]
            relevant_files.sort()

            if relevant_files:
                # 并行读取所有文件内容（优化）
                async def read_file(file: str) -> str:
                    file_path = character_dir_path / file
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            return await f.read()
                    except Exception:
                        return f"[Error reading file: {file}]"

                # 使用 asyncio.gather 并行读取所有文件
                file_contents = await asyncio.gather(*[
                    read_file(file) for file in relevant_files
                ])

                # 拼接文件内容（分隔符：---）
                character_diary_content = "\n\n---\n\n".join(file_contents)

        except FileNotFoundError:
            # 目录不存在，返回默认内容
            character_diary_content = f'[无法读取"{character_name}"的日记本，可能不存在]'
        except Exception as char_dir_error:
            # 其他错误
            print(f'[RAGDiaryPlugin] Error reading character directory {character_dir_path}: {char_dir_error}')
            character_diary_content = f'[无法读取"{character_name}"的日记本，可能不存在]'

        return character_diary_content

    def _sigmoid(self, x: float) -> float:
        """
        Sigmoid 激活函数：将数值映射到 0~1 区间
        :param x: 输入数值
        :return: Sigmoid 计算结果
        """
        return 1 / (1 + math.exp(-x))

    def _truncate_core_tags(self, tags: List[str], ratio: float, metrics: Dict[str, float]) -> List[str]:
        """
        截断核心标签列表
        :param tags: 标签列表
        :param ratio: 截断比例
        :param metrics: 包含 L 和 S 值的指标字典
        :return: 截断后的标签列表
        """
        # 如果标签较少（<=5个），不进行截断，保留原始语义
        if not tags or len(tags) <= 5:
            return tags

        # 动态计算保留数量，最小保留 5 个（除非原始数量不足）
        target_count = max(5, math.ceil(len(tags) * ratio))
        truncated = tags[:target_count]

        if len(truncated) < len(tags):
            logger.info(
                f"[Truncation] {len(tags)} -> {len(truncated)} tags "
                f"(Ratio: {ratio:.2f}, L:{metrics['L']:.2f}, S:{metrics['S']:.2f})"
            )

        return truncated

    # ==================== Phase 1: Core Infrastructure ====================

    async def get_single_embedding(self, text: str) -> Optional[List[float]]:
        """
        获取单个文本的嵌入向量，支持超长文本分块

        Args:
            text: 要向量化的文本

        Returns:
            嵌入向量，失败返回 None
        """
        if not text or not text.strip():
            logger.error("[RAGDiaryPlugin] get_single_embedding called with empty text")
            return None

        if self.vector_db_manager is None:
            logger.error("[RAGDiaryPlugin] vector_db_manager not initialized")
            return None

        try:
            # 获取配置
            async with EmbeddingService() as embedding_service:
                vector = await embedding_service.get_single_embedding(text)
                return vector

        except Exception as e:
            logger.error(f"[RAGDiaryPlugin] Failed to get embedding: {e}")
            return None

    async def _calculate_dynamic_params(
        self,
        query_vector: List[float],
        user_text: str,
        ai_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        V3 动态参数计算：结合逻辑深度 (L)、共振 (R) 和语义宽度 (S)

        1. 基础 K 值计算（基于文本长度）
        2. 使用 sigmoid 计算 beta
        3. Tag 权重从 beta 映射
        4. 最终 K = k_base + L·3 + log(1+R)·2
        5. Tag 截断比例动态计算

        Args:
            query_vector: 查询向量
            user_text: 用户文本
            ai_text: AI 文本（可选）

        Returns:
            包含动态参数的字典:
                - k: 动态 K 值
                - tag_weight: Tag 权重
                - tag_truncation_ratio: Tag 截断比例
                - metrics: {L, R, S, beta} 指标
        """

        # ==================== 1. 基础 K 值计算 (基于文本长度) ====================
        user_len = len(user_text) if user_text else 0
        k_base = 3
        if user_len > 100:
            k_base = 6
        elif user_len > 30:
            k_base = 4

        # 如果有 AI 文本，根据唯一 token 数调整
        if ai_text:
            # 匹配英文单词/数字 或 中文字符
            tokens = re.findall(r'[a-zA-Z0-9]+|[^\s\x00-\xff]', ai_text)
            unique_tokens = set(tokens)
            unique_count = len(unique_tokens)

            if unique_count > 100:
                k_base = max(k_base, 6)
            elif unique_count > 40:
                k_base = max(k_base, 4)

        # ==================== 2. 获取 EPA 指标 (L, R) ====================
        # 使用 vector_db_manager.get_epa_analysis() 统一接口
        L = 0.5  # 逻辑深度
        R = 0.0  # 共振

        if hasattr(self.vector_db_manager, 'get_epa_analysis'):
            epa_analysis = self.vector_db_manager.get_epa_analysis(query_vector)
            L = epa_analysis.get('logic_depth', 0.5)
            R = epa_analysis.get('resonance', 0.0)

        # ==================== 3. 获取语义宽度 (S) ====================
        S = 1.0
        if hasattr(self, 'context_vector_manager'):
            query_np = np.array(query_vector, dtype=np.float32)
            S = self.context_vector_manager.compute_semantic_width(query_np)
            logger.debug(f"[RAGDiary] 📏 Semantic width S={S:.3f}")

        # ==================== 4. 计算动态 Beta (TagWeight) ====================
        # β = σ(L · log(1 + R) - S · noise_penalty)
        noise_penalty = 0.05
        beta_input = L * math.log(1 + R + 1) - S * noise_penalty
        beta = self._sigmoid(beta_input)

        # 将 beta 映射到合理的 RAG 权重范围 [0.05, 0.45]
        weight_range = [0.05, 0.45]
        final_tag_weight = weight_range[0] + beta * (weight_range[1] - weight_range[0])

        # ==================== 5. 计算动态 K ====================
        # 逻辑越深(L)且共振越强(R)，说明信息量越大，需要更高的 K 来覆盖
        k_adjustment = round(L * 3 + math.log1p(R) * 2)
        final_k = max(3, min(10, k_base + k_adjustment))

        # ==================== 6. 计算动态 Tag 截断比例 ====================
        # 逻辑：逻辑越深(L)说明意图越明确，可以保留更多 Tag
        #      语义宽度(S)越大说明噪音或干扰越多，应收紧截断
        # 基础比例 0.6，范围 [0.5, 0.9]
        tag_truncation_ratio = (
            0.6 +
            (L * 0.3) -
            (S * 0.2) +
            (min(R, 1) * 0.1)
        )
        truncation_range = [0.5, 0.9]
        tag_truncation_ratio = max(
            truncation_range[0],
            min(truncation_range[1], tag_truncation_ratio)
        )

        logger.info(
            f"[V3] L={L:.3f}, R={R:.3f}, S={S:.3f} => "
            f"Beta={beta:.3f}, TagWeight={final_tag_weight:.3f}, K={final_k}"
        )

        return {
            'k': final_k,
            'tag_weight': final_tag_weight,
            'tag_truncation_ratio': tag_truncation_ratio,
            'metrics': {'L': L, 'R': R, 'S': S, 'beta': beta}
        }

    # ==================== Phase 2: Text Processing Tools ====================

    def _strip_tool_markers(self, text: str) -> str:
        """
        移除 AI 工具调用的技术标记，防止向量噪音

        保留非黑名单的内容（如日记内容），只过滤技术性字段。

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        if not text or not isinstance(text, str):
            return text

        # 黑名单键名
        blacklisted_keys = ['tool_name', 'command', 'archery', 'maid']
        # 黑名单值
        blacklisted_values = ['dailynote', 'update', 'create', 'no_reply']

        def replace_tool_call_block(match: re.Match) -> str:
            """处理工具调用块，提取并保留有效内容"""
            block = match.group(1)

            results = []

            # 匹配完整的 「始」...「末」 容器
            kv_pattern = r'(\w+):\s*[「『]始[」』]([\s\S]*?)[「『]末[」』]'
            for kv_match in re.finditer(kv_pattern, block):
                key = kv_match.group(1).lower()
                val = kv_match.group(2).strip()
                val_lower = val.lower()

                is_tech_key = key in blacklisted_keys
                is_tech_val = any(bv in val_lower for bv in blacklisted_values)

                # 保留非黑名单的内容
                if not is_tech_key and not is_tech_val and len(val) > 1:
                    results.append(val)

            # 如果正则没匹配到（旧格式或非标准格式），回退到行处理
            if not results:
                lines = []
                for line in block.split('\n'):
                    # 移除键值标记
                    clean_line = re.sub(r'\w+:\s*[「『]始[」』]', '', line)
                    clean_line = re.sub(r'[「『]末[」』]', '', clean_line)
                    clean_line = clean_line.strip()

                    # 过滤包含黑名单值的行
                    line_lower = clean_line.lower()
                    if any(bv in line_lower for bv in blacklisted_values):
                        continue

                    if clean_line:
                        lines.append(clean_line)

                return '\n'.join(lines)

            return '\n'.join(results)

        # 1. 处理工具调用块
        tool_call_pattern = r'<<<\[?TOOL_REQUEST\]?>>>([\s\S]*?)<<<\[?END_TOOL_REQUEST\]?>>>'
        processed = re.sub(tool_call_pattern, replace_tool_call_block, text, flags=re.IGNORECASE)

        # 2. 移除残留的工具调用标记
        processed = re.sub(r'<<<\[?TOOL_REQUEST\]?>>>', '', processed, flags=re.IGNORECASE)
        processed = re.sub(r'<<<\[?END_TOOL_REQUEST\]?>>>', '', processed, flags=re.IGNORECASE)

        # 3. 移除残留的键值标记符号
        processed = re.sub(r'[「」『』]始[「」『』]', '', processed)
        processed = re.sub(r'[「」『』]末[「」『』]', '', processed)
        processed = re.sub(r'[「」『』]', '', processed)

        # 4. 压缩空格（仅压缩水平空格，保留换行）
        processed = re.sub(r'[ \t]+', ' ', processed)

        # 5. 压缩过多换行
        processed = re.sub(r'\n{3,}', '\n\n', processed)

        return processed.strip()

    def _strip_system_notification(self, text: str) -> str:
        """
        移除追加的系统通知内容（净化追加的系统提示框）

        移除类似 "[系统提示: xxx]" 或 "[系统通知: xxx]" 的内容。

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        if not text or not isinstance(text, str):
            return text

        # 移除系统通知/提示块
        processed = re.sub(
            r'\[系统(?:提示|通知):[^\]]*\]',
            '',
            text
        )

        # 移除残留的空格和换行
        processed = re.sub(r'\n{3,}', '\n\n', processed)
        processed = processed.strip()

        return processed

    # ==================== Phase 3: RAG Core Flow ====================

    # 消息的预处理
    async def process_messages(
        self,
        messages: List[Dict[str, Any]],
        plugin_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        处理消息并执行 RAG 检索

        - 更新上下文向量映射
        - 识别需要处理的 system 消息
        - 提取最后一个用户消息和 AI 消息
        - 清理内容（移除系统通知和工具标记）
        - 向量化组合上下文
        - 计算动态参数
        - 获取历史分段和时间范围
        - 处理每个 system 消息

        Args:
            messages: 消息列表
            plugin_config: 插件配置

        Returns:
            处理后的消息列表
        """
        try:
            if not messages:
                return messages

            if self.vector_db_manager is None:
                logger.warning("[RAGDiaryPlugin] vector_db_manager not initialized")
                return messages

            # ✅ 新增：更新上下文向量映射（为后续衰减聚合做准备）
            # 🌟 修复：传递 allowApi 配置，控制是否允许向量化历史消息
            if hasattr(self, 'context_vector_manager'):
                logger.debug("[RAGDiary] 📥 Calling update_context...")
                self.context_vector_manager.update_context(messages, {'allowApi': self.context_vector_allow_api})

            logger.info("[RAGDiaryPlugin] Processing messages for RAG...")

            # 1. 识别需要处理的 system 消息（包含日记本占位符）
            target_system_message_indices = []
            for i, msg in enumerate(messages):
                if msg.get('role') == 'system':
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        # 检查 RAG 日记本占位符
                        if re.search(r'\[\[.*日记本.*\]\]|\{\{.*日记本\}\}', content):
                            target_system_message_indices.append(i)

            # 如果没有找到任何需要处理的 system 消息，则直接返回
            if not target_system_message_indices:
                return messages

            # 2. 准备共享资源 (V3.3: 精准上下文提取)
            # 始终寻找最后一个用户消息和最后一个AI消息，以避免注入污染
            # V3.4: 跳过特殊的 "系统邀请指令" user 消息

            def _extract_text_content(msg: Dict) -> str:
                """提取消息的文本内容"""
                content = msg.get('content', '')
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'text':
                            return item.get('text', '')
                return ''

            # 查找最后一个用户消息（跳过系统邀请指令）
            last_user_message_index = -1
            last_ai_message_index = -1

            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]

                if last_user_message_index == -1 and msg.get('role') == 'user':
                    content = _extract_text_content(msg)
                    # 跳过系统邀请指令
                    if (content and
                        not content.startswith('[系统邀请指令:]') and
                        not content.strip().startswith('[系统提示:]无内容')):
                        last_user_message_index = i

                if last_ai_message_index == -1 and msg.get('role') == 'assistant':
                    # AI 消息不需要特殊过滤
                    if _extract_text_content(msg):
                        last_ai_message_index = i

                if last_user_message_index != -1 and last_ai_message_index != -1:
                    break

            user_content = ''
            ai_content = None

            if last_user_message_index > -1:
                user_content = _extract_text_content(messages[last_user_message_index])

            if last_ai_message_index > -1:
                ai_content = _extract_text_content(messages[last_ai_message_index])

            # V3.1: 在向量化之前，清理userContent和aiContent中的系统通知和工具标记
            if user_content:
                original_user_content = user_content
                user_content = self._strip_system_notification(user_content)  # 净化追加的系统提示框
                user_content = self._strip_tool_markers(user_content)  # 净化工具调用噪音
                if len(original_user_content) != len(user_content):
                    logger.info('[RAGDiaryPlugin] User content was sanitized.')

            if ai_content:
                original_ai_content = ai_content
                ai_content = self._strip_tool_markers(ai_content)  # 净化工具调用噪音
                if len(original_ai_content) != len(ai_content):
                    logger.info('[RAGDiaryPlugin] AI content was sanitized.')

            # V3.5: 为 VCP Info 创建一个更清晰的组合查询字符串
            combined_query_for_display = (
                f'[AI]: {ai_content}\n[User]: {user_content}'
                if ai_content else user_content
            )

            logger.info('[RAGDiaryPlugin] 对完整上下文进行统一向量化...')
            # ✅ 关键修复：不再分开向量化再平均，而是直接对合并后的上下文进行向量化
            query_vector = await self.get_single_embedding(combined_query_for_display)

            if not query_vector:
                # 检查是否是系统提示导致的空内容（这是正常情况）
                is_system_prompt = not user_content or len(user_content) == 0
                if is_system_prompt:
                    logger.info('[RAGDiaryPlugin] 检测到系统提示消息，无需向量化，跳过RAG处理。')
                else:
                    logger.error('[RAGDiaryPlugin] 查询向量化失败，跳过RAG处理。')
                    logger.error(f'[RAGDiaryPlugin] userContent length: {len(user_content)}')
                    logger.error(f'[RAGDiaryPlugin] aiContent length: {len(ai_content) if ai_content else 0}')

                # 安全起见，移除所有占位符
                new_messages = copy.deepcopy(messages)
                for index in target_system_message_indices:
                    if isinstance(new_messages[index].get('content'), str):
                        new_messages[index]['content'] = re.sub(
                            r'\[\[.*日记本.*\]\]', '', new_messages[index]['content']
                        )
                return new_messages

            # 🌟 V3 增强：计算动态参数 (K, TagWeight)
            dynamic_params = await self._calculate_dynamic_params(query_vector, user_content, ai_content)

            # 🌟 Tagmemo V4: 获取上下文分段 (Segments)
            history_segments = []
            if hasattr(self, 'context_vector_manager'):
                try:
                    history_segments = self.context_vector_manager.segment_context(messages)
                    if history_segments:
                        logger.info(f'[RAGDiaryPlugin] Tagmemo V4: Detected {len(history_segments)} history segments.')
                except Exception as e:
                    logger.warning(f'[RAGDiaryPlugin] Context segmentation failed: {e}')

            # 解析时间范围
            combined_text_for_time_parsing = '\n'.join(
                [user_content, ai_content] if ai_content else [user_content]
            )
            time_ranges = self.time_parser.parse(combined_text_for_time_parsing)

            # 🌟 V4.1: 上下文日记去重 - 提取当前上下文中所有 DailyNote create 的 Content 前缀
            context_diary_prefixes = self._extract_context_diary_prefixes(messages)

            # 3. 循环处理每个识别到的 system 消息
            new_messages = copy.deepcopy(messages)
            global_processed_diaries = set()  # 在最外层维护一个 Set

            for index in target_system_message_indices:
                logger.info(f'[RAGDiaryPlugin] Processing system message at index: {index}')

                try:
                    processed_content = await self._process_single_system_message(
                        content=new_messages[index].get('content', ''),
                        query_vector=query_vector,
                        user_content=user_content,
                        ai_content=ai_content,
                        combined_query_for_display=combined_query_for_display,
                        dynamic_k=dynamic_params['k'],
                        time_ranges=time_ranges,
                        processed_diaries=global_processed_diaries,
                        dynamic_tag_weight=dynamic_params['tag_weight'],
                        tag_truncation_ratio=dynamic_params['tag_truncation_ratio'],
                        metrics=dynamic_params['metrics'],
                        history_segments=history_segments,
                        context_diary_prefixes=context_diary_prefixes
                    )

                    new_messages[index]['content'] = processed_content

                except Exception as e:
                    logger.error(f'[RAGDiaryPlugin] Failed to process system message at index {index}: {e}')
                    import traceback
                    logger.error(f'[RAGDiaryPlugin] Traceback: {traceback.format_exc()}')

            return new_messages

        except Exception as error:
            logger.error('[RAGDiaryPlugin] process_messages 发生严重错误:')
            logger.error(f'[RAGDiaryPlugin] Error: {error}')
            import traceback
            logger.error(f'[RAGDiaryPlugin] Traceback: {traceback.format_exc()}')

            # 返回原始消息，移除占位符以避免二次错误
            safe_messages = copy.deepcopy(messages)
            for msg in safe_messages:
                if msg.get('role') == 'system' and isinstance(msg.get('content'), str):
                    msg['content'] = re.sub(
                        r'\[\[.*日记本.*\]\]|\{\{.*日记本\}\}',
                        '[RAG处理失败]',
                        msg['content']
                    )
            return safe_messages

    async def _process_single_system_message(
        self,
        content: str,
        query_vector: List[float],
        user_content: str,
        ai_content: Optional[str],
        combined_query_for_display: str,
        dynamic_k: int,
        time_ranges: List[TimeRange],
        processed_diaries: Set[str],
        dynamic_tag_weight: float = 0.15,
        tag_truncation_ratio: float = 0.5,
        metrics: Dict[str, float] = None,
        history_segments: List[Dict] = None,
        context_diary_prefixes: Set[str] = None
    ) -> str:
        """
        处理单个系统消息中的 RAG 占位符

        - 处理 [[...]] 中的 RAG 请求
        - 处理 {{...日记本}} 直接引入模式
        - 使用 processed_diaries 防止循环引用
        - 并行处理所有请求

        Args:
            content: 消息内容
            query_vector: 查询向量
            user_content: 用户内容
            ai_content: AI 内容
            combined_query_for_display: 组合查询（用于显示）
            dynamic_k: 动态 K 值
            time_ranges: 时间范围列表
            processed_diaries: 已处理的日记集合
            dynamic_tag_weight: 动态 Tag 权重
            tag_truncation_ratio: Tag 截断比例
            metrics: 指标字典
            history_segments: 历史分段
            context_diary_prefixes: 上下文日记前缀

        Returns:
            处理后的内容
        """
        if metrics is None:
            metrics = {'L': 0.5, 'R': 0.0, 'S': 1.0, 'beta': 1.0}
        if history_segments is None:
            history_segments = []
        if context_diary_prefixes is None:
            context_diary_prefixes = set()

        processed_content = content

        # 1. 识别占位符：[[...日记本...]] 和 {{...日记本}}
        rag_declarations = re.findall(r'\[\[(.*?)日记本(.*?)\]\]', content)
        direct_diaries_declarations = re.findall(r'\{\{(.*?)日记本\}\}', content)

        logger.info(f"[RAGDiary] 🔍 识别到 {len(rag_declarations)} 个 RAG 占位符, {len(direct_diaries_declarations)} 个直接引入占位符")
        for db_name, modifiers in rag_declarations:
            logger.info(f"[RAGDiary]   - RAG占位符: [[{db_name}日记本{modifiers}]]")

        processing_promises = []

        # --- 1. 处理 [[...]] 中的 RAG 请求 ---
        for db_name, modifiers in rag_declarations:
            placeholder = f'[[{db_name}日记本{modifiers}]]'
            logger.info(f"[RAGDiary] 🚀 开始处理 RAG 占位符: {placeholder}")

            if db_name in processed_diaries:
                logger.warning(f"[RAGDiaryPlugin] Detected circular reference to \"{db_name}\" in [[...]]. Skipping.")
                processing_promises.append(asyncio.coroutine(lambda: (placeholder, '[检测到循环引用，已跳过"' + db_name + '"日记本"的解析]'))())
                continue

            processed_diaries.add(db_name)

            # 标准 RAG 立即处理
            async def process_rag():
                try:
                    retrieved_content = await self._process_rag_placeholder(
                        db_name=db_name,
                        modifiers=modifiers,
                        query_vector=query_vector,
                        user_content=user_content,
                        combined_query_for_display=combined_query_for_display,
                        dynamic_k=dynamic_k,
                        time_ranges=time_ranges,
                        allow_time_and_group=True,
                        default_tag_weight=dynamic_tag_weight,
                        tag_truncation_ratio=tag_truncation_ratio,
                        metrics=metrics,
                        history_segments=history_segments,
                        context_diary_prefixes=context_diary_prefixes
                    )
                    return (placeholder, retrieved_content)
                except Exception as e:
                    logger.error(f"[RAGDiaryPlugin] 处理占位符时出错 ({db_name}): {e}")
                    return (placeholder, f'[处理失败: {str(e)}]')

            processing_promises.append(process_rag())

        # --- 2. 处理 {{...日记本}} 直接引入模式 ---
        for db_name in direct_diaries_declarations:
            placeholder = f'{{{{{db_name}日记本}}}}'

            if db_name in processed_diaries:
                logger.warning(f"[RAGDiaryPlugin] Detected circular reference to \"{db_name}\" in {{...}}. Skipping.")
                processing_promises.append(asyncio.coroutine(lambda: (placeholder, '[检测到循环引用，已跳过"' + db_name + '"日记本"的解析]'))())
                continue

            # 标记以防其他模式循环
            processed_diaries.add(db_name)

            # 直接获取内容，跳过阈值判断
            async def process_direct():
                try:
                    diary_content = await self._get_diary_content(db_name)
                    # 移除循环占位符
                    safe_content = re.sub(r'\[\[.*日记本.*\]\]', '[循环占位符已移除]', diary_content)
                    safe_content = re.sub(r'\{\{.*日记本\}\}', '[循环占位符已移除]', safe_content)

                    logger.info(f"[RAGDiary] 📄 直接引入日记本：{db_name}")

                    return (placeholder, safe_content)
                except Exception as e:
                    logger.error(f"[RAGDiaryPlugin] 处理 {{...日记本}} 直接引入模式出错 ({db_name}): {e}")
                    return (placeholder, f'[处理失败: {str(e)}]')

            processing_promises.append(process_direct())

        # --- 3. 执行所有任务并替换内容 ---
        if processing_promises:
            logger.info(f"[RAGDiary] ⏳ 开始执行 {len(processing_promises)} 个处理任务...")
            results = await asyncio.gather(*processing_promises, return_exceptions=True)

            logger.info(f"[RAGDiary] ✅ 处理任务完成，开始替换占位符...")
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"[RAGDiaryPlugin] Task failed: {result}")
                    continue

                if isinstance(result, tuple) and len(result) == 2:
                    placeholder, replacement_content = result
                    logger.info(f"[RAGDiary]   替换占位符: {placeholder[:50]}...")
                    processed_content = processed_content.replace(placeholder, replacement_content)

        return processed_content

    async def _process_rag_placeholder(
        self,
        db_name: str,
        modifiers: str,
        query_vector: List[float],
        user_content: str,
        combined_query_for_display: str,
        dynamic_k: int,
        time_ranges: List[TimeRange],
        allow_time_and_group: bool = True,
        default_tag_weight: float = 0.15,
        tag_truncation_ratio: float = 0.5,
        metrics: Dict[str, float] = None,
        history_segments: List[Dict] = None,
        context_diary_prefixes: Set[str] = None
    ) -> str:
        """
        处理 RAG 占位符的核心逻辑

        Args:
            db_name: 数据库名称
            modifiers: 修饰符字符串
            query_vector: 查询向量
            user_content: 用户内容
            combined_query_for_display: 组合查询（用于显示）
            dynamic_k: 动态 K 值
            time_ranges: 时间范围列表
            allow_time_and_group: 是否允许时间和分组
            default_tag_weight: 默认 Tag 权重
            tag_truncation_ratio: Tag 截断比例
            metrics: 指标字典
            history_segments: 历史分段
            context_diary_prefixes: 上下文日记前缀

        Returns:
            格式化的检索结果
        """
        if metrics is None:
            metrics = {'L': 0.5, 'R': 0.0, 'S': 1.0, 'beta': 1.0}
        if history_segments is None:
            history_segments = []
        if context_diary_prefixes is None:
            context_diary_prefixes = set()

        logger.info(f"[RAGDiary] 🔎 开始执行检索: db_name={db_name}, k={k}, use_time={use_time}, use_rerank={use_rerank}, tag_weight={tag_weight}")

        # 1. 解析修饰符
        use_time = False
        use_rerank = False
        use_tag_memo = False
        custom_k = None
        tag_weight = None

        if modifiers:
            use_time = 'Time' in modifiers
            use_rerank = 'Rerank' in modifiers
            use_tag_memo = 'TagMemo' in modifiers

            # 提取自定义 K 值 (格式: ::K10)
            k_match = re.search(r'K(\d+)', modifiers)
            if k_match:
                custom_k = int(k_match.group(1))

            # 提取 TagMemo 权重 (格式: ::TagMemo0.3)
            tag_memo_match = re.search(r'TagMemo([\d.]+)', modifiers)
            if tag_memo_match:
                tag_weight = float(tag_memo_match.group(1))
            elif use_tag_memo:
                tag_weight = default_tag_weight

        # 使用 K 值乘数调整动态 K
        k_multiplier = self._extract_k_multiplier(modifiers)
        base_k = custom_k if custom_k is not None else dynamic_k
        k = max(1, round(base_k * k_multiplier))

        # 去重缓冲 (V4.1: 补偿去重损失)
        dedup_buffer = len(context_diary_prefixes)
        k_for_search = k + dedup_buffer
        if use_rerank and hasattr(self, 'rerank_config') and self.rerank_config:
            rerank_multiplier = self.rerank_config.get('multiplier', 1.5)
            k_for_search = int(k * rerank_multiplier) + dedup_buffer

        logger.debug(f"[RAGDiary] 📊 K值计算: base_k={base_k:.1f}, multiplier={k_multiplier:.2f}, final_k={k}, k_for_search={k_for_search}")

        # 2. 原子级复刻 LightMemo 流程：利用 applyTagBoost 预先感应语义 Tag
        core_tags: List[str] = []

        if tag_weight is not None and tag_weight > 0 and self.vector_db_manager:
            try:
                query_np = np.array(query_vector, dtype=np.float32)
                tag_boost_result = self.vector_db_manager.apply_tag_boost(
                    vector=query_np,
                    tag_boost=tag_weight,
                    core_tags=core_tags if core_tags else []
                )
                if tag_boost_result and tag_boost_result.info and 'matched_tags' in tag_boost_result.info:
                    raw_tags = tag_boost_result.info['matched_tags']
                    # 应用截断技术规避尾部噪音
                    core_tags = self._truncate_core_tags(raw_tags, tag_truncation_ratio, metrics)
                    logger.info(f"[RAGDiaryPlugin] 感应到核心 Tag: [{', '.join(core_tags)}] "
                               f"{'(从 ' + str(len(raw_tags)) + ' 个截断)' if len(raw_tags) > len(core_tags) else ''}")

            except Exception as e:
                logger.warning(f"[RAGDiaryPlugin] Tag boost failed: {e}")

        # 3. 执行检索
        final_results = []
        logger.info(f"[RAGDiary] 🔎 开始执行检索: db_name={db_name}, k={k}, use_time={use_time}, use_rerank={use_rerank}, tag_weight={tag_weight}")

        if use_time and time_ranges and allow_time_and_group:
            # --- 平衡双路召回 (Balanced Dual-Path Retrieval) ---
            # 目标：语义召回占 60%，时间召回占 40%，且时间召回也进行相关性排序
            k_semantic = max(1, math.ceil(k * 0.6))
            k_time = max(1, k - k_semantic)

            logger.info(f"[RAGDiaryPlugin] Time-Aware Balanced Mode: Total K={k} (Semantic={k_semantic}, Time={k_time})")

            # 1. 语义路召回
            rag_results = []
            if self.vector_db_manager:
                try:
                    logger.info(f"[RAGDiary] 📊 语义路召回: 搜索 {k_for_search} 条...")
                    rag_results = await self.vector_db_manager.search(
                        db_name,
                        query_vector,
                        k_for_search,
                        tag_weight if tag_weight else 0.0,
                        core_tags if core_tags else None
                    )
                    logger.info(f"[RAGDiary] ✅ 语义路召回完成: 获取 {len(rag_results)} 条结果")
                    rag_results = self._filter_context_duplicates(rag_results, context_diary_prefixes)
                    rag_results = rag_results[:k_semantic]
                    # 添加 source 标识
                    for r in rag_results:
                        r.__dict__['source'] = 'rag'
                    logger.info(f"[RAGDiary] 📊 语义路去重后: {len(rag_results)} 条")

                    # 显示语义路召回的文件路径
                    for i, r in enumerate(rag_results[:10]):
                        file_path = getattr(r, 'full_path', '') or getattr(r, 'source_file', '')
                        file_name = Path(file_path).name if file_path else 'unknown'
                        score = getattr(r, 'score', 0)
                        logger.debug(f"[RAGDiary]   Semantic[{i}]: score={score:.2f}, file={file_name}")
                    if len(rag_results) > 10:
                        logger.debug(f"[RAGDiary]   ... and {len(rag_results) - 10} more results")
                except Exception as e:
                    logger.warning(f"[RAGDiaryPlugin] Semantic search failed: {e}")

            # 2. 时间路召回 (带相关性排序)
            # 收集所有时间范围的文件路径
            time_file_paths = []
            for time_range in time_ranges:
                try:
                    files = await self._get_time_range_file_paths(db_name, time_range)
                    time_file_paths.extend(files)
                except Exception as e:
                    logger.warning(f"[RAGDiaryPlugin] Time range file paths failed: {e}")

            # 去重文件路径
            time_file_paths = list(set(time_file_paths))

            time_results = []
            if time_file_paths and self.vector_db_manager:
                try:
                    logger.info(f"[RAGDiary] 📅 时间路召回: 找到 {len(time_file_paths)} 个文件")
                    # 从数据库获取这些文件的所有分块及其向量
                    time_chunks = await self.vector_db_manager.get_chunks_by_file_paths(time_file_paths)

                    # 计算每个分块与当前查询向量的相似度
                    for chunk in time_chunks:
                        vector = getattr(chunk, 'vector', None)
                        if vector:
                            chunk.score = self.cosine_similarity(query_vector, vector)
                        else:
                            chunk.score = 0.0
                        chunk.__dict__['source'] = 'time'

                    # 按相似度排序并取前 k_time 个
                    time_chunks.sort(key=lambda c: c.score, reverse=True)
                    time_results = time_chunks[:k_time]
                    logger.info(f"[RAGDiaryPlugin] Time path: Found {len(time_chunks)} chunks in range, selected top {len(time_results)} by relevance.")
                except Exception as e:
                    logger.warning(f"[RAGDiaryPlugin] Time chunks processing failed: {e}")

            # 去重时间路结果
            time_results = self._filter_context_duplicates(time_results, context_diary_prefixes)

            # 3. 合并与去重
            all_entries = {}
            # 语义路优先
            for r in rag_results:
                key = r.text.strip() if hasattr(r, 'text') else str(r)
                all_entries[key] = r
            # 时间路补充（如果内容不重复）
            for r in time_results:
                key = r.text.strip() if hasattr(r, 'text') else str(r)
                if key not in all_entries:
                    all_entries[key] = r

            final_results = list(all_entries.values())
            logger.info(f"[RAGDiary] 🎯 双路合并结果: {len(final_results)} 条 (语义: {len(rag_results)}, 时间: {len(time_results)})")

            # 显示合并后的文件路径
            for i, r in enumerate(final_results[:10]):
                file_path = getattr(r, 'full_path', '') or getattr(r, 'source_file', '')
                file_name = Path(file_path).name if file_path else 'unknown'
                source = getattr(r, 'source', 'unknown')
                score = getattr(r, 'score', 0)
                logger.debug(f"[RAGDiary]   Merged[{i}]: score={score:.2f}, source={source}, file={file_name}")
            if len(final_results) > 10:
                logger.debug(f"[RAGDiary]   ... and {len(final_results) - 10} more results")

            # 如果启用了 Rerank，对合并后的结果进行最终重排
            if use_rerank and final_results:
                try:
                    final_results = await self._rerank_documents(user_content, final_results, k)
                except Exception as e:
                    logger.warning(f"[RAGDiaryPlugin] Rerank failed: {e}")

        else:
            # --- 标准路径：霰弹枪查询 (Shotgun Query) ---
            # 使用当前查询向量 + 历史 3 个分段向量进行并行搜索

            search_vectors = [{'vector': query_vector, 'type': 'current', 'weight': 1.0}]

            # 仅在存在历史分段且未使用 Time 模式时启用霰弹枪
            if history_segments and len(history_segments) > 0:
                # 限制: 最多取最近的 3 个分段
                recent_segments = history_segments[-3:]

                # V5.1: 时间距离衰减惩罚 (Decay Multiplier)
                # 距离越远（index 越小），权重越低
                decay_factor = 0.85

                for idx, seg in enumerate(recent_segments):
                    distance = len(recent_segments) - idx
                    weight_multiplier = math.pow(decay_factor, distance)

                    if 'vector' in seg:
                        search_vectors.append({
                            'vector': seg['vector'],
                            'type': f'history_{idx}',
                            'weight': weight_multiplier
                        })

            logger.info(f"[RAGDiaryPlugin] Shotgun Query: Executing {len(search_vectors)} parallel searches with decay weights")

            # 并行搜索
            logger.info(f"[RAGDiary] 🔫 霰弹枪模式: {len(search_vectors)} 个向量并行搜索...")
            search_tasks = []
            for sv in search_vectors:
                sv_k = k_for_search if sv['type'] == 'current' else max(2, k_for_search // 2)

                async def search_with_weight(sv_info, k_val):
                    try:
                        results = await self.vector_db_manager.search(
                            db_name,
                            sv_info['vector'],
                            k_val,
                            tag_weight if tag_weight else 0.0,
                            core_tags if core_tags else None
                        )
                        # 应用时间权重衰减
                        if sv_info['weight'] != 1.0:
                            for r in results:
                                r.__dict__['original_score'] = r.score
                                r.score = r.score * sv_info['weight']
                        return results
                    except Exception as e:
                        logger.warning(f"[RAGDiaryPlugin] Shotgun search failed for {sv_info['type']}: {e}")
                        return []

                search_tasks.append(search_with_weight(sv, sv_k))

            results_arrays = await asyncio.gather(*search_tasks)
            flattened_results = []
            for arr in results_arrays:
                flattened_results.extend(arr)

            logger.info(f"[RAGDiary] ✅ 霰弹枪搜索完成: 原始 {len(flattened_results)} 条结果")

            # 上下文去重
            flattened_results = self._filter_context_duplicates(flattened_results, context_diary_prefixes)

            # SVD 智能去重
            # 使用 vector_db_manager.deduplicate_results() 统一接口
            if flattened_results and hasattr(self.vector_db_manager, 'deduplicate_results'):
                try:
                    flattened_results = await self.vector_db_manager.deduplicate_results(flattened_results, query_vector)
                except Exception as e:
                    logger.warning(f"[RAGDiaryPlugin] Deduplication failed: {e}")

            # Rerank 或截断
            if use_rerank and flattened_results:
                try:
                    final_results = await self._rerank_documents(user_content, flattened_results, k)
                except Exception as e:
                    logger.warning(f"[RAGDiaryPlugin] Rerank failed: {e}")
                    final_results = flattened_results[:k]
            else:
                final_results = flattened_results[:k]

            # 添加 source 标识
            for r in final_results:
                r.__dict__['source'] = 'rag'

        logger.info(f"[RAGDiary] 📋 最终结果: {len(final_results)} 条，准备格式化...")

        # 4. 输出检索日志
        self._log_rag_results(
            db_name, k, tag_weight, use_time, use_rerank, core_tags,
            combined_query_for_display, time_ranges, final_results
        )

        # 5. 格式化结果
        if use_time and time_ranges:
            return self.format_combined_time_aware_results(
                final_results[:k],
                time_ranges,
                db_name,
                {'query': combined_query_for_display, 'k': k, 'core_tags': core_tags}
            )
        else:
            return self.format_standard_results(
                final_results[:k],
                db_name,
                {'query': combined_query_for_display, 'k': k, 'core_tags': core_tags}
            )

    def _log_rag_results(
        self,
        db_name: str,
        k: int,
        tag_weight: Optional[float],
        use_time: bool,
        use_rerank: bool,
        core_tags: List[str],
        combined_query: str,
        time_ranges: List[TimeRange],
        results: List
    ) -> None:
        """输出 RAG 检索日志"""
        try:
            # 按相关度分数排序
            sorted_results = sorted(
                results,
                key=lambda r: getattr(r, 'rerank_score', None) or r.score or -1,
                reverse=True
            )

            display_results = sorted_results[:5]

            logger.info('')
            logger.info('═══════════════════════════════════════════════════════════')
            logger.info(f'[RAG检索] {db_name}日记本')
            logger.info('═══════════════════════════════════════════════════════════')

            # 参数信息
            logger.info('📊 参数:')
            logger.info(f'   - K: {k}')
            if tag_weight is not None:
                logger.info(f'   - TagWeight: {tag_weight:.3f}')
            logger.info(f'   - Time Mode: {"是" if use_time else "否"}')
            logger.info(f'   - Rerank: {"是" if use_rerank else "否"}')
            if core_tags:
                logger.info(f'   - CoreTags: [{", ".join(core_tags)}]')

            # 查询内容
            query_preview = combined_query[:150] + '...' if len(combined_query) > 150 else combined_query
            logger.info(f'\n📝 查询:\n   {query_preview}')

            # 时间范围
            if use_time and time_ranges:
                logger.info('\n📅 时间范围:')
                for tr in time_ranges:
                    start = tr.start.strftime('%Y-%m-%d') if tr.start else '?'
                    end = tr.end.strftime('%Y-%m-%d') if tr.end else '?'
                    logger.info(f'   - {start} ~ {end}')

            # 检索结果
            logger.info(f'\n🎯 检索结果 (Top {len(display_results)}/{len(results)}):')
            for i, r in enumerate(display_results):
                score = getattr(r, 'rerank_score', None) or r.score or 0
                score_pct = score * 100
                source = getattr(r, 'source', 'unknown')

                # 获取文件路径信息
                file_path = getattr(r, 'full_path', '') or getattr(r, 'source_file', '')
                if file_path:
                    logger.info(f'\n   [{i+1}] 相似度: {score_pct:.1f}% | 来源: {source} | 文件: {file_path}')
                else:
                    logger.info(f'\n   [{i+1}] 相似度: {score_pct:.1f}% | 来源: {source}')

                matched_tags = getattr(r, 'matched_tags', None)
                if matched_tags:
                    logger.info(f'       Tags: [{", ".join(matched_tags)}]')

                text = getattr(r, 'text', '')
                text_preview = text[:80].replace('\n', ' ') + '...' if len(text) > 80 else text.replace('\n', ' ')
                logger.info(f'       {text_preview}')

            # Tag 统计
            if tag_weight is not None:
                tag_stats = self._aggregate_tag_stats(results)
                if tag_stats and tag_stats.get('uniqueMatchedTags'):
                    logger.info(f'\n🏷️  Tag统计:')
                    logger.info(f'   - 唯一标签: [{", ".join(tag_stats["uniqueMatchedTags"])}]')
                    logger.info(f'   - 匹配数: {tag_stats["totalTagMatches"]}')
                    logger.info(f'   - 有Tag的结果: {tag_stats["resultsWithTags"]}/{len(results)}')

            logger.info('═══════════════════════════════════════════════════════════')
            logger.info('')

        except Exception as e:
            logger.error(f'[RAGDiaryPlugin] 日志输出失败: {e}')

    # ==================== Phase 4: Time-aware Retrieval ====================

    async def _get_time_range_file_paths(
        self,
        db_name: str,
        time_range: TimeRange
    ) -> List[str]:
        """
        获取时间范围内的文件路径列表

        - 直接读取文件系统（不使用数据库）
        - 只读取每个文件的前 100 字符
        - 从首行提取日期格式 [2026.03.01] 或 [2026-03-01]
        - 返回相对路径 dbName/filename

        Args:
            db_name: 数据库名称（角色名）
            time_range: 时间范围

        Returns:
            相对文件路径列表 (dbName/filename 格式)
        """
        file_paths_in_range: List[str] = []

        # 检查时间范围
        if not time_range or not time_range.start or not time_range.end:
            return file_paths_in_range

        try:
            # 构建角色日记目录路径
            character_dir_path = dailyNoteRootPath / db_name

            if not character_dir_path.exists():
                return file_paths_in_range

            # 读取目录文件列表
            files = os.listdir(character_dir_path)
            diary_files = [
                f for f in files
                if f.lower().endswith(('.txt', '.md'))
            ]

            # 日期正则：匹配 [2026.03.01] 或 [2026-03-01] 格式
            date_pattern = re.compile(r'^\[?(\d{4}[-.]\d{2}[-.]\d{2})\]?')

            for file in diary_files:
                file_path = character_dir_path / file

                try:
                    # 只读取前 100 字符
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read(100)

                    # 获取第一行
                    first_line = content.split('\n')[0] if '\n' in content else content

                    # 提取日期
                    match = date_pattern.search(first_line)
                    if match:
                        date_str = match.group(1)
                        # 标准化日期：将 . 替换为 -
                        normalized_date_str = date_str.replace('.', '-')

                        # 解析日期并设置为时区的开始时间（与 JS 版本一致：保留在本地时区）
                        from datetime import datetime as dt
                        import pytz
                        tz = pytz.timezone(DEFAULT_TIMEZONE)

                        try:
                            # 先在本地时区解析，然后设置为当天开始（00:00:00），保留在本地时区
                            diary_date = tz.localize(
                                dt.strptime(normalized_date_str, '%Y-%m-%d')
                            ).replace(hour=0, minute=0, second=0, microsecond=0)

                            # 检查是否在时间范围内
                            if time_range.start <= diary_date <= time_range.end:
                                # 返回相对路径：dbName/filename
                                relative_path = f'{db_name}/{file}'
                                file_paths_in_range.append(relative_path)

                        except ValueError as e:
                            logger.debug(f"[RAGDiaryPlugin] Failed to parse date {normalized_date_str}: {e}")

                except Exception:
                    # 单个文件读取失败不影响其他文件
                    pass

        except Exception as dir_error:
            logger.error(f"[RAGDiaryPlugin] Failed to read directory {db_name}: {dir_error}")

        return file_paths_in_range


    # ==================== Phase 6: Context Deduplication ====================

    def _extract_context_diary_prefixes(
        self,
        messages: List[Dict]
    ) -> Set[str]:
        """
        提取上下文日记前缀（V4.1 去重）

        从 AI 助手消息的工具调用块中提取 DailyNote create 命令的 content 字段前缀。

        Args:
            messages: 消息列表

        Returns:
            日记内容前缀集合（每个前缀 80 字符）
        """
        PREFIX_LEN = 80
        prefixes: Set[str] = set()

        for msg in messages:
            # 仅处理 assistant 消息
            if msg.get('role') != 'assistant':
                continue

            # 获取内容
            content = msg.get('content', '')
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        content = item.get('text', '')
                        break
                else:
                    content = ''

            if not isinstance(content, str):
                continue

            # 跳过不包含 TOOL_REQUEST 的消息
            if 'TOOL_REQUEST' not in content:
                continue

            # 匹配所有工具调用块
            block_pattern = r'<<<\[?TOOL_REQUEST\]?>>>([\s\S]*?)<<<\[?END_TOOL_REQUEST\]?>>>'
            for block_match in re.finditer(block_pattern, content, re.IGNORECASE):
                block = block_match.group(1)

                # 提取键值对（「始」...「末」格式）
                kv_pattern = r'(\w+):\s*[「『]始[」』]([\s\S]*?)[「『]末[」』]'
                fields = {}
                for kv_match in re.finditer(kv_pattern, block):
                    key = kv_match.group(1).lower()
                    value = kv_match.group(2).strip()
                    fields[key] = value

                # 仅处理 DailyNote create 指令
                tool_name = fields.get('tool_name', '').lower()
                command = fields.get('command', '').lower()

                if tool_name == 'dailynote' and command == 'create':
                    content_field = fields.get('content', '')
                    if content_field:
                        prefix = content_field[:PREFIX_LEN].strip()
                        if prefix:
                            prefixes.add(prefix)

        if prefixes:
            logger.info(f"[ContextDedup] Extracted {len(prefixes)} diary prefixes from context")

        return prefixes

    def _filter_context_duplicates(
        self,
        results: List[Dict],
        prefixes: Set[str]
    ) -> List[Dict]:
        """
        过滤已在上下文中的召回结果（V4.1 上下文日记去重）

        通过比较结果文本内容的前 80 字符与上下文前缀来检测重复。

        Args:
            results: 检索结果列表（包含 text 字段）
            prefixes: 上下文日记前缀集合

        Returns:
            过滤后的结果列表
        """
        if not prefixes or not results:
            return results

        PREFIX_LEN = 80
        MIN_COMPARE_LEN = 10

        filtered = []

        for result in results:
            text = _get_attr(result, 'text', '')

            if not text:
                # 没有文本内容，保留
                filtered.append(result)
                continue

            # 日记条目格式: "[2026-02-15] - 角色名\n[14:00] 内容..."
            # 需要跳过日期头 "[yyyy-MM-dd] - name\n" 来匹配 Content 字段
            body = text.strip()

            # 匹配并移除日期头
            header_pattern = r'^\[\d{4}-\d{2}-\d{2}\]\s*-\s*.*?\n'
            header_match = re.match(header_pattern, body)
            if header_match:
                body = body[header_match.end():]

            # 取前 80 字符作为结果前缀
            result_prefix = body[:PREFIX_LEN].strip()

            if not result_prefix:
                # 空前缀，保留
                filtered.append(result)
                continue

            # 检查是否与任一上下文前缀匹配
            is_duplicate = False
            for ctx_prefix in prefixes:
                # 取两者较短长度进行比较
                compare_len = min(len(result_prefix), len(ctx_prefix))

                if compare_len > MIN_COMPARE_LEN:
                    # 前缀匹配：检查前缀是否相同
                    if result_prefix[:compare_len] == ctx_prefix[:compare_len]:
                        is_duplicate = True
                        break

            if not is_duplicate:
                filtered.append(result)

        removed_count = len(results) - len(filtered)
        if removed_count > 0:
            logger.info(f"[ContextDedup] Filtered {removed_count} duplicates from context")

        return filtered

    # ==================== Helper Functions ====================

    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的 Token 数量（中英文混合）

        使用启发式算法：
        - 中文字符：约 1.5 tokens/char
        - 英文字符：约 0.25 tokens/char (1 word ≈ 4 chars)

        Args:
            text: 待估算的文本

        Returns:
            估算的 Token 数量
        """
        if not text:
            return 0

        # 匹配中文字符（Unicode 范围：\u4e00-\u9fa5）
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        other_chars = len(text) - chinese_chars

        # 中文: ~1.5 token/char, 英文: ~0.25 token/char
        estimated_tokens = math.ceil(chinese_chars * 1.5 + other_chars * 0.25)

        return estimated_tokens

    def _extract_k_multiplier(self, modifiers: str) -> float:
        """
        从修饰符字符串中提取 K 值乘数

        Args:
            modifiers: 修饰符字符串，如 "Time:2.5"

        Returns:
            K 值乘数，默认 1.0

        Examples:
            >>> _extract_k_multiplier("Time:2.5")
            2.5
            >>> _extract_k_multiplier("Rerank")
            1.0
        """
        if not modifiers or not isinstance(modifiers, str):
            return 1.0

        # 匹配冒号后的数字（支持小数）
        match = re.search(r':(\d+\.?\d*)', modifiers)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return 1.0

        return 1.0

    def _aggregate_tag_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """
        聚合标签统计信息

        用于日志记录或 VCP Info 广播

        Args:
            results: 检索结果列表（应包含 matched_tags 和 boost_factor 字段）

        Returns:
            统计信息字典:
                - unique_matched_tags: 唯一匹配标签列表
                - total_tag_matches: 唯一标签数量
                - results_with_tags: 有标签的结果数量
                - avg_boost_factor: 平均增强因子
        """
        all_matched_tags = set()
        total_boost_factor = 0.0
        results_with_tags = 0

        for result in results:
            matched_tags = _get_attr(result, 'matched_tags', [])
            if matched_tags and len(matched_tags) > 0:
                for tag in matched_tags:
                    all_matched_tags.add(tag)
                results_with_tags += 1

                boost_factor = _get_attr(result, 'boost_factor', 0.0)
                total_boost_factor += boost_factor

        avg_boost_factor = (
            round(total_boost_factor / results_with_tags, 3)
            if results_with_tags > 0
            else 1.0
        )

        return {
            'unique_matched_tags': list(all_matched_tags),
            'total_tag_matches': len(all_matched_tags),
            'results_with_tags': results_with_tags,
            'avg_boost_factor': avg_boost_factor
        }

    # ==================== Rerank Support ====================

    async def _rerank_documents(
        self,
        query: str,
        documents: List[Dict],
        original_k: int
    ) -> List[Dict]:
        """
        使用 Rerank API 重新排序文档

        - 断路器模式防止频繁调用失败的 API
        - Token 感知查询截断
        - 智能批处理
        - 详细错误处理和提前终止

        Args:
            query: 查询文本
            documents: 文档列表
            original_k: 原始 K 值

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return documents

        # ==================== JIT 配置检查 ====================
        rerank_url = self.rerank_config.get('url', '')
        rerank_api_key = self.rerank_config.get('api_key', '')
        rerank_model = self.rerank_config.get('model', '')

        if not rerank_url or not rerank_api_key or not rerank_model:
            logger.warning('[RAGDiaryPlugin] Rerank called, but is not configured. Skipping.')
            return documents[:original_k]

        # ==================== 断路器模式 ====================
        if not hasattr(self, '_rerank_circuit_breaker'):
            self._rerank_circuit_breaker = {}

        import time
        now = time.time()

        # 检查1分钟内的失败次数
        recent_failures = sum(
            1 for timestamp in self._rerank_circuit_breaker.values()
            if now - timestamp < 60  # 1分钟内
        )

        if recent_failures >= 5:
            logger.warning('[RAGDiaryPlugin] Rerank circuit breaker activated due to recent failures. Skipping rerank.')
            return documents[:original_k]

        # ==================== 查询截断机制 ====================
        max_query_tokens = int(self.rerank_config.get('max_tokens', 30000) * 0.3)
        query_tokens = self._estimate_tokens(query)
        truncated_query = query

        if query_tokens > max_query_tokens:
            logger.warning(f'[RAGDiaryPlugin] Query too long ({query_tokens} tokens), truncating to {max_query_tokens} tokens')
            truncate_ratio = max_query_tokens / query_tokens
            target_length = int(len(query) * truncate_ratio * 0.9)  # 留10%安全边距
            truncated_query = query[:target_length] + '...'
            query_tokens = self._estimate_tokens(truncated_query)
            logger.info(f'[RAGDiaryPlugin] Query truncated to {query_tokens} tokens')

        # ==================== 准备 Rerank URL ====================
        # 确保 URL 格式正确
        if not rerank_url.endswith('/'):
            rerank_url += '/'
        rerank_url = f"{rerank_url}v1/rerank"

        max_tokens = self.rerank_config.get('max_tokens', 30000)

        # ==================== 智能批处理逻辑 ====================
        batches = []
        current_batch = []
        current_tokens = query_tokens
        min_batch_size = 1
        max_batch_tokens = max_tokens - query_tokens - 1000  # 预留1000 tokens安全边距

        for doc in documents:
            doc_text = _get_attr(doc, 'text', '')
            doc_tokens = self._estimate_tokens(doc_text)

            # 如果单个文档就超过限制，跳过该文档
            if doc_tokens > max_batch_tokens:
                logger.warning(f'[RAGDiaryPlugin] Document too large ({doc_tokens} tokens), skipping')
                continue

            if current_tokens + doc_tokens > max_batch_tokens and len(current_batch) >= min_batch_size:
                # 当前批次已满，保存并开始新批次
                batches.append(current_batch)
                current_batch = [doc]
                current_tokens = query_tokens + doc_tokens
            else:
                # 添加到当前批次
                current_batch.append(doc)
                current_tokens += doc_tokens

        # 添加最后一个批次
        if current_batch:
            batches.append(current_batch)

        # 如果没有有效批次，直接返回原始文档
        if not batches:
            logger.warning('[RAGDiaryPlugin] No valid batches for reranking, returning original documents')
            return documents[:original_k]

        logger.info(f'[RAGDiaryPlugin] Rerank processing {len(batches)} batches with truncated query ({query_tokens} tokens)')

        # ==================== 处理批次 ====================
        import httpx
        all_reranked_docs = []
        failed_batches = 0

        for i, batch in enumerate(batches):
            doc_texts = [_get_attr(d, 'text', '') for d in batch]

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        rerank_url,
                        headers={
                            'Authorization': f'Bearer {rerank_api_key}',
                            'Content-Type': 'application/json'
                        },
                        json={
                            'model': rerank_model,
                            'query': truncated_query,
                            'documents': doc_texts,
                            'top_n': len(doc_texts)
                        }
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if 'results' in data and isinstance(data['results'], list):
                            reranked_results = data['results']
                            ordered_batch = []
                            for result in reranked_results:
                                idx = result.get('index', 0)
                                if idx < len(batch):
                                    original_doc = batch[idx]
                                    ordered_batch.append({
                                        **original_doc,
                                        'rerank_score': result.get('relevance_score', 0.0)
                                    })
                            all_reranked_docs.extend(ordered_batch)
                        else:
                            logger.warning(f'[RAGDiaryPlugin] Rerank for batch {i + 1} returned invalid data. Appending original batch documents.')
                            all_reranked_docs.extend(batch)
                            failed_batches += 1
                    else:
                        logger.warning(f'[RAGDiaryPlugin] Rerank API returned status {response.status_code} for batch {i + 1}')
                        all_reranked_docs.extend(batch)
                        failed_batches += 1

            except httpx.TimeoutException:
                failed_batches += 1
                logger.error('[RAGDiaryPlugin] Rerank API timeout')
                self._rerank_circuit_breaker[f'rerank_{int(now)}_{i}'] = now
                all_reranked_docs.extend(batch)

                # 提前终止检查
                if i > 2 and failed_batches / (i + 1) > 0.5:
                    logger.warning('[RAGDiaryPlugin] Too many rerank failures, terminating early')
                    # 添加剩余批次的原始文档
                    for j in range(i + 1, len(batches)):
                        all_reranked_docs.extend(batches[j])
                    break

            except httpx.HTTPStatusError as e:
                failed_batches += 1
                status = e.response.status_code
                logger.error(f'[RAGDiaryPlugin] Rerank API Error - Status: {status}')

                # 特定错误处理
                if status == 400:
                    error_data = e.response.json() if hasattr(e.response, 'json') else {}
                    error_message = error_data.get('error', {}).get('message', '')
                    if 'Query is too long' in error_message:
                        logger.error('[RAGDiaryPlugin] Query still too long after truncation, adding to circuit breaker')
                        self._rerank_circuit_breaker[f'rerank_{int(now)}_{i}'] = now
                elif status >= 500:
                    # 服务器错误，添加到断路器
                    self._rerank_circuit_breaker[f'rerank_{int(now)}_{i}'] = now

                all_reranked_docs.extend(batch)

                # 提前终止检查
                if i > 2 and failed_batches / (i + 1) > 0.5:
                    logger.warning('[RAGDiaryPlugin] Too many rerank failures, terminating early')
                    for j in range(i + 1, len(batches)):
                        all_reranked_docs.extend(batches[j])
                    break

            except Exception as e:
                failed_batches += 1
                logger.error(f'[RAGDiaryPlugin] Rerank API Error - Message: {str(e)}')
                self._rerank_circuit_breaker[f'rerank_{int(now)}_{i}'] = now
                all_reranked_docs.extend(batch)

                # 提前终止检查
                if i > 2 and failed_batches / (i + 1) > 0.5:
                    logger.warning('[RAGDiaryPlugin] Too many rerank failures, terminating early')
                    for j in range(i + 1, len(batches)):
                        all_reranked_docs.extend(batches[j])
                    break

        # ==================== 清理过期断路器记录 ====================
        expired_keys = [
            key for key, timestamp in self._rerank_circuit_breaker.items()
            if now - timestamp > 300  # 5分钟后清理
        ]
        for key in expired_keys:
            del self._rerank_circuit_breaker[key]

        # ==================== 全局排序 ====================
        all_reranked_docs.sort(key=lambda x: x.get('rerank_score', x.get('score', -1)), reverse=True)

        final_docs = all_reranked_docs[:original_k]
        success_rate = ((len(batches) - failed_batches) / len(batches) * 100) if batches else 0
        logger.info(f'[RAGDiaryPlugin] Rerank完成: {len(final_docs)}篇文档 (成功率: {success_rate:.1f}%)')

        return final_docs

    # ==================== Result Formatting ====================

    def format_standard_results(
        self,
        search_results: List[Dict],
        display_name: str,
        metadata: Dict
    ) -> str:
        """

        Args:
            search_results: 搜索结果列表
            display_name: 显示名称
            metadata: 元数据

        Returns:
            格式的文本
        """
        logger.info(f"[RAGDiary] 📝 格式化标准结果: {len(search_results)} 条 -> \"{display_name}\"")

        # 构建内部内容
        inner_content = f'\n[--- 从"{display_name}"中检索到的相关记忆片段 ---]\n'

        if search_results:
            for i, result in enumerate(search_results[:metadata.get('k', 10)]):
                text = _get_attr(result, 'text', '').strip()
                logger.debug(f"[RAGDiary]   结果[{i+1}]: {text[:50]}...")
                inner_content += f'* {text}\n'
        else:
            inner_content += '没有找到直接相关的记忆片段。'

        inner_content += '\n[--- 记忆片段结束 ---]\n'

        # 转义元数据中的 -->
        metadata_string = json.dumps(metadata, ensure_ascii=False).replace('-->', '--\\>')

        result = f'<!-- RAG_BLOCK_START {metadata_string} -->{inner_content}<!-- RAG_BLOCK_END -->'
        logger.info(f"[RAGDiary] ✅ 格式化完成，结果长度: {len(result)} 字符")
        return result

    def format_combined_time_aware_results(
        self,
        results: List[Dict],
        time_ranges: List[TimeRange],
        db_name: str,
        metadata: Dict
    ) -> str:
        """
        格式化时间感知结果为 RAG_BLOCK 格式（多时间感知）

        - 分离语义相关和时间范围结果
        - 添加统计信息
        - 分别显示两个章节

        Args:
            results: 结果列表（包含 source 字段区分 'rag'/'time'）
            time_ranges: 时间范围列表
            db_name: 数据库名称
            metadata: 元数据

        Returns:
            RAG_BLOCK 格式的文本
        """
        logger.info(f"[RAGDiary] 📝 格式化时间感知结果: {len(results)} 条 -> \"{db_name}日记本\"")

        # 显示名称
        display_name = f'{db_name}日记本'

        # 日期格式化函数
        def format_date(dt: datetime) -> str:
            return dt.strftime('%Y-%m-%d')

        # 构建内部内容
        inner_content = f'\n[--- "{display_name}" 多时间感知检索结果 ---]\n'

        # 格式化时间范围
        formatted_ranges = ' 和 '.join([
            f'"{format_date(r.start)} ~ {format_date(r.end)}"'
            for r in time_ranges
        ])
        inner_content += f'[合并查询的时间范围: {formatted_ranges}]\n'

        # 分离结果为语义相关和时间范围
        rag_entries = [e for e in results if _get_attr(e, 'source') == 'rag']
        time_entries = [e for e in results if _get_attr(e, 'source') == 'time']

        # 添加统计信息
        inner_content += f'[统计: 共找到 {len(results)} 条不重复记忆 (语义相关 {len(rag_entries)}条, 时间范围 {len(time_entries)}条)]\n\n'

        # 语义相关记忆章节
        if rag_entries:
            inner_content += '【语义相关记忆】\n'
            for entry in rag_entries:
                text = _get_attr(entry, 'text', '')
                # 获取文件路径
                file_path = _get_attr(entry, 'full_path', '') or _get_attr(entry, 'source_file', '')
                # 提取日期前缀
                date_match = re.match(r'^\[(\d{4}-\d{2}-\d{2})\]', text)
                date_prefix = f'[{date_match.group(1)}] ' if date_match else ''
                # 移除日期头
                cleaned_text = re.sub(r'^\[.*?\]\s*-\s*.*?\n?', '', text).strip()
                # 添加文件路径信息
                path_info = f' 📁 {Path(file_path).name}' if file_path else ''
                inner_content += f'* {date_prefix}{cleaned_text}{path_info}\n'

        # 时间范围记忆章节
        if time_entries:
            inner_content += '\n【时间范围记忆】\n'
            # 按日期从新到旧排序
            sorted_time_entries = sorted(
                time_entries,
                key=lambda e: _get_attr(e, 'date', ''),
                reverse=True
            )
            for entry in sorted_time_entries:
                text = _get_attr(entry, 'text', '')
                # 移除日期头
                cleaned_text = re.sub(r'^\[.*?\]\s*-\s*.*?\n?', '', text).strip()
                date_str = _get_attr(entry, 'date', '')
                inner_content += f'* [{date_str}] {cleaned_text}\n'

        inner_content += '[--- 检索结束 ---]\n'

        # 转义元数据中的 -->
        metadata_string = json.dumps(metadata, ensure_ascii=False).replace('-->', '--\\>')

        return f'<!-- _RAG_BLOCK_START {metadata_string} -->{inner_content}<!-- _RAG_BLOCK_END -->'

    def _clean_results_for_broadcast(self, results: List[Dict]) -> List[Dict]:
        """
        清理结果用于广播/日志记录，仅保留可序列化的关键属性

        Args:
            results: 原始结果列表

        Returns:
            清理后的结果列表，仅包含可序列化的属性
        """
        if not results or not isinstance(results, list):
            return []

        cleaned_results = []

        for r in results:
            # 仅保留可序列化的关键属性
            cleaned = {
                'text': _get_attr(r, 'text', ''),
                'score': _get_attr(r, 'score'),
                'source': _get_attr(r, 'source'),
                'date': _get_attr(r, 'date'),
            }

            # 包含 Tag 相关信息（如果存在）
            # 安全地检查动态属性（适用于 dataclass 和普通对象）
            try:
                if hasattr(r, '__dict__'):
                    r_dict = r.__dict__
                    if 'originalScore' in r_dict:
                        cleaned['originalScore'] = r_dict['originalScore']
                    if 'tagMatchScore' in r_dict:
                        cleaned['tagMatchScore'] = r_dict['tagMatchScore']
                    if 'tagMatchCount' in r_dict:
                        cleaned['tagMatchCount'] = r_dict['tagMatchCount']
            except (TypeError, AttributeError):
                pass  # r 是普通字典或其他类型，跳过动态属性检查

            # matchedTags 使用 dataclass 属性 matched_tags
            matched_tags = _get_attr(r, 'matched_tags', [])
            if matched_tags:
                cleaned['matchedTags'] = matched_tags

            # boostFactor 使用 dataclass 属性 boost_factor
            boost_factor = _get_attr(r, 'boost_factor', 0.0)
            if boost_factor != 0.0:
                cleaned['boostFactor'] = boost_factor

            # coreTagsMatched 使用 dataclass 属性 core_tags_matched
            core_tags = _get_attr(r, 'core_tags_matched', [])
            if core_tags:
                cleaned['coreTagsMatched'] = [
                    t for t in core_tags
                    if isinstance(t, str)
                ]

            cleaned_results.append(cleaned)

        return cleaned_results


# ==================== Module-level exports for plugin manager ====================

# 创建全局插件实例
_plugin_instance: Optional['RAGDiaryPlugin'] = None


def initialize(config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
    """
    初始化插件（模块级别，供 PluginManager 调用）

    Args:
        config: 插件配置字典
        dependencies: 依赖注入字典 (vectorDBManager 等)
    """
    global _plugin_instance
    if _plugin_instance is None:
        _plugin_instance = RAGDiaryPlugin()
        _plugin_instance.rag_config = config
        _plugin_instance.vector_db_manager = dependencies.get('vectorDBManager')
        _plugin_instance.is_initialized = True
        logger.info("[RAGDiaryPlugin] Plugin initialized via module-level initialize()")


async def process_messages(messages: List[Dict[str, Any]], plugin_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    处理消息（模块级别，供 PluginManager 调用）

    Args:
        messages: 消息列表
        plugin_config: 插件配置

    Returns:
        处理后的消息列表
    """
    global _plugin_instance

    if _plugin_instance is None:
        logger.warning("[RAGDiaryPlugin] Plugin not initialized, creating instance on-the-fly")
        _plugin_instance = RAGDiaryPlugin()
        _plugin_instance.is_initialized = True

    return await _plugin_instance.process_messages(messages, plugin_config)


def shutdown() -> None:
    """关闭插件（模块级别）"""
    global _plugin_instance
    _plugin_instance = None
    logger.info("[RAGDiaryPlugin] Plugin shutdown")
