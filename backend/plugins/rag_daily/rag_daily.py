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
# 日记文件根路径：从项目根目录的 data/daily/ 读取
dailyNoteRootPath = Path(__file__).parent.parent.parent.parent / "data" / "daily"


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

    def _strip_tool_markers(self, text):
        """
        清理工具调用标记文本，过滤黑名单关键词，提取有效内容
        """
        if not text or not isinstance(text, str):
            return text

        # 黑名单配置
        blacklisted_keys = {'tool_name', 'command', 'archery', 'maid'}
        blacklisted_values = {'dailynote', 'update', 'create', 'no_reply'}

        # 1. 匹配并处理工具调用块 <<<[TOOL_REQUEST]>>> ... <<<[END_TOOL_REQUEST]>>>
        def replace_tool_block(match):
            block = match.group(1)
            results = []

            # 2. 匹配 key:「始」value「末」格式（支持「」/『』）
            import re
            regex = re.compile(r'(\w+):\s*[「『]始[」』]([\s\S]*?)[「『]末[」』]', re.DOTALL)
            for m in regex.finditer(block):
                key = m.group(1).lower()
                val = m.group(2).strip()
                val_lower = val.lower()

                # 黑名单过滤
                is_tech_key = key in blacklisted_keys
                is_tech_val = any(bv in val_lower for bv in blacklisted_values)

                if not is_tech_key and not is_tech_val and len(val) > 1:
                    results.append(val)

            # 正则未匹配到 → 回退到按行处理（兼容旧格式）
            if not results:
                lines = block.split('\n')
                cleaned_lines = []
                for line in lines:
                    # 清理标记
                    clean_line = re.sub(r'\w+:\s*[「『]始[」』]', '', line)
                    clean_line = re.sub(r'[「『]末[」』]', '', clean_line).strip()
                    lower_line = clean_line.lower()
                    # 黑名单过滤
                    if any(bv in lower_line for bv in blacklisted_values):
                        continue
                    if clean_line:
                        cleaned_lines.append(clean_line)
                return '\n'.join(cleaned_lines)

            return '\n'.join(results)

        # 执行全局替换（不区分大小写）
        processed = re.sub(
            r'<<<\[?TOOL_REQUEST\]?>>>([\s\S]*?)<<<\[?END_TOOL_REQUEST\]?>>>',
            replace_tool_block,
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # 3. 清理残余标记、符号、多余空格和换行
        processed = re.sub(r'<<<\[?TOOL_REQUEST\]?>>>', '', processed, flags=re.IGNORECASE)
        processed = re.sub(r'<<<\[?END_TOOL_REQUEST\]?>>>', '', processed, flags=re.IGNORECASE)
        processed = re.sub(r'[「」『』]始[「」『』]', '', processed)
        processed = re.sub(r'[「」『』]末[「」『』]', '', processed)
        processed = re.sub(r'[「」『』]', '', processed)
        processed = re.sub(r'[ \t]+', ' ', processed)  # 压缩水平空格
        processed = re.sub(r'\n{3,}', '\n\n', processed)  # 压缩过多换行
        processed = processed.strip()

        return processed
    
    # ==================== Phase 3: RAG Core Flow ====================

    async def process_messages(
        self,
        messages: List[Dict[str, Any]],
        plugin_config: Optional[Dict[str, Any]] = None,
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

            await self.context_vector_manager.update_context(
                messages, 
                {'allowApi': self.context_vector_allow_api}
            )

            logger.info("[RAGDiaryPlugin] Processing messages for RAG...")

            # 1. 识别需要处理的 system 消息（包含日记本占位符）
            import re  # 确保 re 模块可用
            target_system_message_indices = []
            for i, msg in enumerate(messages):
                if msg.get('role') == 'system':
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        # 检查 RAG 日记本占位符
                        if re.search(r'\[\[.*日记本.*\]\]', content):
                            target_system_message_indices.append(i)

            # 如果没有找到任何需要处理的 system 消息，则直接返回
            if not target_system_message_indices:
                logger.warning("[RAGDiaryPlugin] ⚠️ No system message with [[日记本]] placeholder found!")
                # 打印所有 system 消息的内容帮助调试
                for i, msg in enumerate(messages):
                    if msg.get('role') == 'system':
                        content = msg.get('content', '')
                        logger.warning(f"[RAGDiaryPlugin]   System message {i}: {content[:200]}...")
                return messages

            last_user_message_index = -1
            last_ai_message_index = -1

            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
            
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        # 查找 text 类型的内容
                        text_part = next((p for p in content if p.get('type') == 'text'), None)
                        text = text_part.get('text', '') if text_part else ''
                    else:
                        text = ''

                    if not text.startswith('[系统邀请指令:]') and not text.startswith('[系统提示:]'):
                        last_user_message_index = i
                        break

            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get('role') == 'assistant':
                    last_ai_message_index = i
                    break

            user_content = ''
            ai_content = None

            if last_user_message_index > -1:
                last_user_message = messages[last_user_message_index]
                content = last_user_message.get('content', '')
                if isinstance(content, str):
                    user_content = content
                elif isinstance(content, list):
                    text_part = next((p for p in content if p.get('type') == 'text'), None)
                    user_content = text_part.get('text', '') if text_part else ''
                else:
                    user_content = ''

            if last_ai_message_index > -1:
                last_ai_message = messages[last_ai_message_index]
                content = last_ai_message.get('content', '')
                if isinstance(content, str):
                    ai_content = content
                elif isinstance(content, list):
                    text_part = next((p for p in content if p.get('type') == 'text'), None)
                    ai_content = text_part.get('text', '') if text_part else ''
                else:
                    ai_content = ''

            # V3.1: 在向量化之前，清理userContent和aiContent中的系统通知和工具标记
            if user_content:
                original_user_content = user_content
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

            user_vector = await self.get_single_embedding(combined_query_for_display)
            ai_vector = (
                await self.get_single_embedding(ai_content)
                if ai_content and getattr(self, 'context_vector_allow_api', False)
                else None
            )

            aggregated_ai_vector = self.context_vector_manager.aggregate_context('assistant')
            aggregated_user_vector = self.context_vector_manager.aggregate_context('user')

            query_vector = None

            if ai_vector and user_vector:
                # 结合当前意图与历史聚合意图
                current_intent = self._get_weighted_average_vector(
                    [user_vector, ai_vector],
                    [0.7, 0.3]
                )
                if aggregated_ai_vector or aggregated_user_vector:
                    history_vectors = []
                    history_weights = []
                    if aggregated_user_vector:
                        history_vectors.append(aggregated_user_vector)
                        history_weights.append(0.6)
                    if aggregated_ai_vector:
                        history_vectors.append(aggregated_ai_vector)
                        history_weights.append(0.4)

                    history_intent = self._get_weighted_average_vector(
                        history_vectors,
                        history_weights
                    )
                    query_vector = self._get_weighted_average_vector(
                        [current_intent, history_intent],
                        [0.8, 0.2]
                    )
                else:
                    query_vector = current_intent
            else:
                query_vector = user_vector or ai_vector

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

            dynamic_params = await self._calculate_dynamic_params(query_vector, user_content, ai_content)

            # 解析时间范围
            combined_text_for_time_parsing = '\n'.join(
                filter(None, [user_content, ai_content])
            )
            time_ranges = self.time_parser.parse(combined_text_for_time_parsing)

            # 3. 循环处理每个识别到的 system 消息
            new_messages = copy.deepcopy(messages)
            global_processed_diaries = set()  # 在最外层维护一个 Set

            for index in target_system_message_indices:
                logger.info(f'[RAGDiaryPlugin] Processing system message at index: {index}')

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
                )

                new_messages[index]['content'] = processed_content

            # 调试：打印返回的消息内容
            for i, msg in enumerate(new_messages):
                if msg.get('role') == 'system':
                    content = msg.get('content', '')
                    has_rag_block = 'RAG_BLOCK_START' in content
                    logger.info(f"[RAGDiaryPlugin] 📤 Returning system message {i}: has_rag_block={has_rag_block}, content_length={len(content)}")

            return new_messages

        except Exception as error:
            logger.error('[RAGDiaryPlugin] process_messages 发生严重错误:')
            import traceback
            logger.error(f'[RAGDiaryPlugin] Traceback: {traceback.format_exc()}')

            # 返回原始消息，移除占位符以避免二次错误
            safe_messages = copy.deepcopy(messages)
            return safe_messages

    async def _process_single_system_message(
        self,
        content: str,
        query_vector: List[float],
        user_content: str,
        ai_content: str,
        combined_query_for_display: str,
        dynamic_k: int,
        time_ranges: List[TimeRange],
        processed_diaries: Set[str],
        dynamic_tag_weight: float = 0.15,
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
            ai_content: 用于rerank
            combined_query_for_display: 组合查询（用于显示）
            dynamic_k: 动态 K 值
            time_ranges: 时间范围列表
            processed_diaries: 已处理的日记集合
            dynamic_tag_weight: 动态 Tag 权重

        Returns:
            处理后的内容
        """

        processed_content = content

        rag_matches = list(re.finditer(r'\[\[(.*?)日记本(.*?)\]\]', content))
        processing_task = []

        for match in rag_matches:
            placeholder = match.group(0)
            db_name = match.group(1)
            modifiers = match.group(2)

            if db_name in processed_diaries:
                logger.warning(f"[RAGDiaryPlugin] Detected circular reference to \"{db_name}\" in {{...}}. Skipping.")
                continue

            processed_diaries.add(db_name)

            # 直接获取内容，跳过阈值判断
            async def process_placeholder(ph=placeholder, db=db_name, mod=modifiers):
                try:
                    retrieved_content = await self._process_rag_placeholder(
                        db_name=db,
                        modifiers=mod,
                        query_vector=query_vector,
                        user_content=user_content,
                        ai_content=ai_content,  # 传递 ai_content
                        dynamic_k=dynamic_k,
                        time_ranges=time_ranges,
                        default_tag_weight=dynamic_tag_weight
                    )
                    return (ph, retrieved_content)
                except Exception as e:
                    logger.error(f"[RAGDiaryPlugin] 处理 处理占位符时出错 ({db}): {e}")
                    return (placeholder, f'[处理失败: {str(e)}]')

            processing_task.append(process_placeholder())

        # --- 执行所有任务并替换内容 ---
        if processing_task:
            results = await asyncio.gather(*processing_task, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f'[RAGDiaryPlugin] Task failed: {result}')
                    continue

                placeholder, replacement_content = result
                processed_content = processed_content.replace(placeholder, replacement_content)

        return processed_content

    async def _process_rag_placeholder(
        self,
        db_name: str,
        modifiers: str,
        query_vector: List[float],
        user_content: str,
        ai_content: str = "",
        dynamic_k: int = 5,
        time_ranges: Optional[List[TimeRange]] = None,
        default_tag_weight: float = 0.15,
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
            default_tag_weight: 默认 Tag 权重

        Returns:
            格式化的检索结果
        """

        # 1. 解析修饰符
        k_multiplier = self._extract_k_multiplier(modifiers)
        use_time = '::Time' in modifiers
        use_rerank = '::Rerank' in modifiers
        tag_memo_match = re.search(r'::TagMemo([\d.]+)', modifiers)
        if tag_memo_match:
            tag_weight = float(tag_memo_match.group(1))
        elif '::TagMemo' in modifiers:
            tag_weight = default_tag_weight
        else:
            tag_weight = None

        display_name = db_name + '日记本'
        final_k = max(1, round(dynamic_k * k_multiplier))
        k_for_search = (
            max(1, round(final_k * self.rerank_config['multiplier']))
            if use_rerank
            else final_k
        )

        metadata = {
          'dbName': db_name,
          'modifiers': modifiers,
          'k': final_k
        }


        if use_time and time_ranges:
            # --- 平衡双路召回 (Balanced Dual-Path Retrieval) ---
            # 目标：语义召回占 60%，时间召回占 40%，且时间召回也进行相关性排序
            # 1. 语义路召回（限定在时间范围内）
            rag_results = await self.vector_db_manager.search(
                db_name, query_vector, k_for_search, tag_weight
            )

            all_entries = {}
            for entry in rag_results:
              key = entry.get('text', '').strip()
              if key and key not in all_entries:
                  all_entries[key] = {**entry, 'source': 'rag'}

            # 添加时间范围结果
            for time_range in time_ranges:
                time_results = await self.get_time_range_diaries(db_name, time_range)
                for entry in time_results:
                    key = entry.get('text', '').strip()
                    if key and key not in all_entries:
                        all_entries[key] = entry

            final_results = list(all_entries.values())
            retrieved_content = self.format_combined_time_aware_results(
                final_results, time_ranges, db_name, metadata
            )

        else:
            search_results = await self.vector_db_manager.search(
                db_name, query_vector, k_for_search, tag_weight
            )

            if use_rerank:
                search_results = await self._rerank_documents(
                    user_content, search_results, final_k
                )

            final_results = [{**vars(r), 'source': 'rag'} for r in search_results]
            retrieved_content = self.format_standard_results(
                search_results, display_name, metadata
            )
            
        return retrieved_content


    # ==================== Phase 4: Time-aware Retrieval ====================

    async def get_time_range_diaries(
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

        diaries_in_range: List[str] = []

        # 检查时间范围
        if not time_range or not time_range.start or not time_range.end:
            logger.warning(f"[RAGDiary] ⚠️ 时间范围无效，返回空列表")
            return diaries_in_range

        try:
            character_dir_path = dailyNoteRootPath / db_name

            # 读取目录文件列表
            files = os.listdir(character_dir_path)
            diary_files = [
                f for f in files
                if f.lower().endswith(('.txt', '.md'))
            ]

            for file in diary_files:
                file_path = character_dir_path / file

                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()

                    first_line = content.split('\n')[0] if '\n' in content else content
                    match = re.match(r'^\[?(\d{4}[-.]\d{2}[-.]\d{2})\]?', first_line)
                    if match:
                        date_str = match.group(1)
                        # 标准化日期：将 . 替换为 -
                        normalized_date_str = date_str.replace('.', '-')

                        import pytz
                        tz = pytz.timezone(DEFAULT_TIMEZONE)

                        date_obj = datetime.strptime(normalized_date_str, '%Y-%m-%d')

                        diary_date = tz.localize(date_obj).replace(
                          hour=0, minute=0, second=0, microsecond=0
                        )
                        if time_range.start <= diary_date <= time_range.end:
                            diaries_in_range.append({
                                'date': normalized_date_str,
                                'text': content,
                                'source': 'time'
                            })

                except Exception:
                    # 单个文件读取失败不影响其他文件
                    pass

        except Exception as dir_error:
            if getattr(dir_error, 'errno', None) != 2: 
                logger.error(
                    f'[RAGDiaryPlugin] Error reading character directory '
                    f'for time filter {character_dir_path}: {dir_error}'
                )

        return diaries_in_range

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

        result = f'<!-- RAG_BLOCK_START {metadata_string} -->{inner_content}<!-- RAG_BLOCK_END -->'
        logger.info(f"[RAGDiary] ✅ 格式化完成，结果长度: {len(result)} 字符")
        return result


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
