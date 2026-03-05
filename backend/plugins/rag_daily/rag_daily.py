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

import json
import math
import aiofiles
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from .time_parser import TimeExpressionParser, TimeRange
from .context_vector_manager import ContextVectorManager
from .epa_module import EPAModule
from .residual_pyramid import ResidualPyramid
from .result_deduplicator import ResultDeduplicator

from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


DEFAULT_TIMEZONE = "Asia/Shanghai"
dailyNoteRootPath = Path(__file__).parent / "daily_notes"


import json
import os
from pathlib import Path
from typing import Dict, Optional, Any

DEFAULT_TIMEZONE = "Asia/Shanghai"


class RAGDiaryPlugin:
    def __init__(self):
        self.name = 'RAGDiaryPlugin'
        self.vector_db_manager: Optional[Any] = None
        self.rag_config: Dict[str, Any] = {}
        self.rerank_config: Dict[str, Any] = {}
        self.push_vcp_info: Optional[callable] = None
        self.time_parser = TimeExpressionParser('zh-CN', DEFAULT_TIMEZONE)
        self.context_vector_manager = ContextVectorManager(self)
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

    def _get_average_vector(self, vectors: List[List[float]]) -> Optional[List[float]]:
        """
        计算多个向量的简单平均向量
        :param vectors: 向量列表
        :return: 平均向量，无有效向量返回None
        """
        if not vectors or len(vectors) == 0:
            return None
        if len(vectors) == 1:
            return vectors[0]
        
        dimension = len(vectors[0])
        result = [0.0] * dimension
        
        # 累加所有向量的对应维度值
        for vec in vectors:
            if not vec or len(vec) != dimension:
                continue
            for i in range(dimension):
                result[i] += vec[i]
        
        # 计算平均值
        vector_count = len(vectors)
        for i in range(dimension):
            result[i] /= vector_count
        
        return result


    async def get_diary_content(self, character_name: str) -> str:
        """
        异步读取指定角色的日记本内容（整合所有 .txt/.md 文件）
        :param character_name: 角色名
        :return: 整合后的日记内容（含错误提示）
        """
        character_dir_path = dailyNoteRootPath / character_name
        character_diary_content = f"[{character_name}日记本内容为空]"
        
        try:
            # 异步读取目录下的文件列表
            # Python 3.10+ 支持 asyncio.scandir，这里用 aiofiles 兼容更多版本
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
                # 异步读取所有文件内容
                file_contents = []
                for file in relevant_files:
                    file_path = character_dir_path / file
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            content = await f.read()
                        file_contents.append(content)
                    except Exception as read_err:
                        file_contents.append(f"[Error reading file: {file}]")
                
                # 拼接文件内容（分隔符：---）
                character_diary_content = "\n\n---\n\n".join(file_contents)

        except Exception as char_dir_error:
            # 仅处理非"目录不存在"的错误（ENOENT 对应 Python 的 FileNotFoundError）
            if not isinstance(char_dir_error, FileNotFoundError):
                print(f'[RAGDiaryPlugin] Error reading character directory {character_dir_path}: {char_dir_error}')
            character_diary_content = f"[无法读取“{character_name}”的日记本，可能不存在]"
        
        return character_diary_content

    def _sigmoid(self, x: float) -> float:
        """
        Sigmoid 激活函数：将数值映射到 0~1 区间
        :param x: 输入数值
        :return: Sigmoid 计算结果
        """
        return 1 / (1 + math.exp(-x))