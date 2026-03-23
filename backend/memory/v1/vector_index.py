import hashlib
import asyncio
import struct
import json
import time
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Union, Set
from dataclasses import dataclass, field
import logging

import numpy as np
from sqlalchemy.orm import Session
from memory.v1.models import DiaryFileTable, ChunkTable, TagTable, FileTagTable, KVStoreTable, SessionLocal
from app.services.chunk_text import chunk_text
from app.services.embedding import EmbeddingService
from app.services.character_service import CharacterService
from memory.v1.plugins.rag_daily.epa_module import EPAModule
from memory.v1.plugins.rag_daily.residual_pyramid import ResidualPyramid

@dataclass
class VectorIndexConfig:
    """向量索引配置"""
    dimension: int = 1024           # 向量维度
    capacity: int = 50000           # 索引容量
    index_save_delay: float = 5.0   # 保存延迟(秒)
    tag_index_save_delay: float = 10.0  # 标签索引保存延迟(秒)

    # 批处理配置
    max_batch_size: int = 20        # 单批处理最大文件数
    batch_delay: float = 2.0        # 批处理延迟(秒)
    max_file_retries: int = 3       # 文件处理最大重试次数

    # 文件监视器配置
    enable_watcher: bool = True     # 是否启用文件监视器
    watch_path: Optional[str] = None  # 监视路径（默认为 data/daily）
    ignore_folders: List[str] = field(default_factory=lambda: ["__pycache__", ".git"])
    ignore_prefixes: List[str] = field(default_factory=lambda: [".", "_"])
    ignore_suffixes: List[str] = field(default_factory=lambda: [".tmp", ".bak", ".swp"])
    allowed_extensions: List[str] = field(default_factory=lambda: [".md", ".txt"])

    # 标签黑名单配置
    tag_blacklist: set = field(default_factory=set)
    tag_blacklist_super: List[str] = field(default_factory=list)

    # 语言置信度补偿
    lang_confidence_enabled: bool = True
    lang_penalty_unknown: float = 0.05
    lang_penalty_cross_domain: float = 0.1

    # Tag 扩展配置
    tag_expand_max_count: int = 30

    # 去重阈值
    deduplication_threshold: float = 0.88
    tech_tag_threshold: float = 0.08
    normal_tag_threshold: float = 0.015

    # 动态增强范围
    activation_multiplier: Tuple[float, float] = (0.5, 1.5)
    dynamic_boost_range: Tuple[float, float] = (0.3, 2.0)
    core_boost_range: Tuple[float, float] = (1.20, 1.40)

    # RAG 参数文件路径
    rag_params_path: Optional[str] = None

    @property
    def store_path(self) -> Path:
        """获取VectorStore目录路径（项目根目录）"""
        # 从 backend/memory/v1/ 向上四级到达项目根目录
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        return project_root / "VectorStore"

    @property
    def data_daily_path(self) -> Path:
        """获取 data/daily 目录路径"""
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        return project_root / "data" / "daily"


from vector_db import VexusIndex


@dataclass
class SearchResult:
    """搜索结果数据类"""
    text: str                          # 匹配的文本内容
    score: float                       # 相似度分数
    source_file: str                   # 来源文件名
    full_path: str = ""                # 完整路径
    updated_at: int = 0                # 更新时间
    matched_tags: List[str] = field(default_factory=list)      # 匹配的标签
    boost_factor: float = 0.0          # 增强因子

@dataclass
class TagBoostResult:
    """Tag 增强结果"""
    vector: List[float]                # 增强后的向量
    info: Optional[Dict[str, Any]]     # 增强信息（匹配的标签等）


class VectorIndex:
    """向量索引管理器 - 支持多索引、Tag 搜索和增强"""

    def __init__(self, config: VectorIndexConfig):
        self.config = config
        self.diary_indices: Dict[str, VexusIndex] = {}  # diaryName -> VexusIndex实例
        self.tag_indices: Dict[str, VexusIndex] = {}    # diaryName -> Tag索引实例（按日记本隔离）
        self.save_tasks: Dict[str, Optional[asyncio.Task]] = {}  # diaryName -> asyncio.Task
        self.tag_save_tasks: Dict[str, Optional[asyncio.Task]] = {}  # diaryName -> asyncio.Task (tag索引保存任务)
        self.diary_name_vector_cache: Dict[str, List[float]] = {}  # 日记本名称向量缓存

        # 批处理队列
        self.pending_files: Set[str] = set()  # 待处理文件集合
        self.file_retry_count: Dict[str, int] = {}  # 文件重试计数
        self.is_processing: bool = False  # 是否正在处理
        self.batch_timer: Optional[asyncio.Task] = None  # 批处理定时器
        self.watcher = None  # watchdog.Observer 实例
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None  # 主事件循环
        self.epa: Optional[EPAModule] = None                    # EPAModule 实例
        self.residual_pyramid: Optional[ResidualPyramid] = None       # ResidualPyramid 实例
        self.tag_cooccurrence_matrix: Dict[int, Dict[int, float]] = {}
        self.rag_params: Dict[str, Any] = {}
        self.rag_params_watcher: Optional[Any] = None     # 文件监视器

        self._ensure_store_path()

    def _ensure_store_path(self) -> None:
        """确保存储目录存在"""
        store_path = self.config.store_path
        if not store_path.exists():
            store_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"[VectorIndex] Created store path: {store_path}")

    async def _init_epa_modules(self) -> None:
        """
        初始化 EPA 模块（按日记本隔离）

        EPA (Embedding Projection Analysis) 模块用于语义空间投影和跨域共振检测
        """
        logging.info("[VectorIndex] 🧠 EPA Modules will be initialized per diary")
        self.epa_modules: Dict[str, EPAModule] = {}  # diary_name -> EPAModule

    async def _get_or_init_epa_module(self, diary_name: str) -> Optional[EPAModule]:
        """
        获取或初始化指定日记本的 EPA 模块

        Args:
            diary_name: 日记本名称

        Returns:
            EPAModule 实例
        """
        if diary_name in self.epa_modules and self.epa_modules[diary_name] and self.epa_modules[diary_name].initialized:
            return self.epa_modules[diary_name]

        try:
            # 先获取 tag 索引
            tag_index = await self._get_or_load_tag_index(diary_name)
            if not tag_index:
                logging.warning(f"[VectorIndex] ⚠️ No tag index for \"{diary_name}\", EPA initialization skipped")
                return None

            # 获取数据库连接
            db_path = str(self.config.store_path / "emotional_companionship.db")
            import sqlite3
            db = sqlite3.connect(db_path, check_same_thread=False)

            epa = EPAModule(
                db=db,
                config={
                    'dimension': self.config.dimension,
                    'max_basis_dim': 64,
                    'min_variance_ratio': 0.01,
                    'cluster_count': 32,
                    'vexus_index': tag_index,
                }
            )

            success = await epa.initialize(diary_name)
            if success:
                self.epa_modules[diary_name] = epa
                logging.info(f"[VectorIndex] ✅ EPA Module initialized for \"{diary_name}\"")
                return epa
            else:
                logging.warning(f"[VectorIndex] ⚠️ EPA Module failed for \"{diary_name}\"")
                return None

        except Exception as e:
            logging.error(f"[VectorIndex] ❌ EPA Module error for \"{diary_name}\": {e}")
            return None

    async def _init_residual_pyramids(self) -> None:
        """
        初始化残差金字塔模块（按日记本隔离）

        残差金字塔用于多级语义残差分析和特征提取
        """
        logging.info("[VectorIndex] 🔺 Residual Pyramid Modules will be initialized per diary")
        self.residual_pyramids: Dict[str, ResidualPyramid] = {}  # diary_name -> ResidualPyramid

    async def _get_or_init_residual_pyramid(self, diary_name: str) -> Optional[ResidualPyramid]:
        """
        获取或初始化指定日记本的残差金字塔模块

        Args:
            diary_name: 日记本名称

        Returns:
            ResidualPyramid 实例
        """
        if diary_name in self.residual_pyramids and self.residual_pyramids[diary_name]:
            return self.residual_pyramids[diary_name]

        try:
            # 先获取 tag 索引
            tag_index = await self._get_or_load_tag_index(diary_name)
            if not tag_index:
                logging.warning(f"[VectorIndex] ⚠️ No tag index for \"{diary_name}\", ResidualPyramid initialization skipped")
                return None

            # 获取数据库连接
            db_path = str(self.config.store_path / "emotional_companionship.db")
            import sqlite3
            db = sqlite3.connect(db_path)

            pyramid = ResidualPyramid(
                tag_index=tag_index,
                db=db,
                config={
                    'dimension': self.config.dimension,
                    'max_levels': 3,
                    'top_k': 10,
                    'min_energy_ratio': 0.1,
                }
            )

            self.residual_pyramids[diary_name] = pyramid
            logging.info(f"[VectorIndex] ✅ Residual Pyramid initialized for \"{diary_name}\"")
            return pyramid

        except Exception as e:
            logging.error(f"[VectorIndex] ❌ Residual Pyramid error for \"{diary_name}\": {e}")
            return None

    async def _build_cooccurrence_matrix(self) -> None:
        """
        构建 Tag 共现矩阵

        从数据库 file_tags 表统计 Tag 共现频率，用于逻辑分支拉回
        """
        logging.info("[VectorIndex] 🧠 Building tag co-occurrence matrix...")

        db_path = str(self.config.store_path / "emotional_companionship.db")
        import sqlite3

        try:
            db = sqlite3.connect(db_path)
            cursor = db.cursor()

            # SQL 查询：统计 Tag 共现频率
            query = """
                SELECT ft1.tag_id as tag1, ft2.tag_id as tag2, COUNT(ft1.file_id) as weight
                FROM file_tags ft1
                JOIN file_tags ft2 ON ft1.file_id = ft2.file_id AND ft1.tag_id < ft2.tag_id
                GROUP BY ft1.tag_id, ft2.tag_id
            """

            cursor.execute(query)
            matrix = {}

            for row in cursor.fetchall():
                tag1, tag2, weight = row
                if tag1 not in matrix:
                    matrix[tag1] = {}
                if tag2 not in matrix:
                    matrix[tag2] = {}
                matrix[tag1][tag2] = float(weight)
                matrix[tag2][tag1] = float(weight)

            self.tag_cooccurrence_matrix = matrix
            logging.info(f"[VectorIndex] ✅ Tag co-occurrence matrix built. ({len(matrix)} tags)")

        except Exception as e:
            logging.error(f"[VectorIndex] ❌ Failed to build co-occurrence matrix: {e}")
            self.tag_cooccurrence_matrix = {}
        finally:
            if 'db' in locals():
                db.close()

    def _apply_language_compensation(
        self,
        tag_name: str,
        query_world: str,
        config: Dict[str, Any]
    ) -> float:
        """
        应用语言置信度补偿

        检测技术噪音标签并应用跨域惩罚

        Args:
            tag_name: 标签名称
            query_world: 查询世界观
            config: RAG 配置参数

        Returns:
            补偿因子（0-1）
        """
        if not self.config.lang_confidence_enabled:
            return 1.0

        # 检测是否为技术噪音
        is_technical_noise = (
            not re.search(r'[\u4e00-\u9fa5]', tag_name) and
            re.match(r'^[A-Za-z0-9\-_.\s]+$', tag_name) and
            len(tag_name) > 3
        )

        is_technical_world = (
            query_world != 'Unknown' and
            re.match(r'^[A-Za-z0-9\-_.]+$', query_world)
        )

        if is_technical_noise and not is_technical_world:
            # 社会世界观检测
            is_social_world = re.search(
                r'Politics|Society|History|Economics|Culture',
                query_world,
                re.IGNORECASE
            )

            # 从 RAG 配置获取惩罚值
            lang_config = config.get('languageCompensator', {})
            base_penalty = (
                lang_config.get('penaltyUnknown', self.config.lang_penalty_unknown)
                if query_world == 'Unknown'
                else lang_config.get('penaltyCrossDomain', self.config.lang_penalty_cross_domain)
            )

            # 社会世界观使用平方根软化惩罚
            return np.sqrt(base_penalty) if is_social_world else base_penalty

        return 1.0

    # ==================== 初始化 ====================
    async def initialize(self) -> None:
        """
        初始化向量索引系统
        - 初始化全局 Tag 索引
        - 预热日记本名称向量缓存
        - 启动文件监视器
        - 初始化 TagMemo V3 模块
        """
        logging.info("[VectorIndex] 🚀 Initializing Multi-Index System...")

        # 保存事件循环引用（用于监视器回调）
        self.event_loop = asyncio.get_running_loop()

        # 1. 初始化日记本 Tag 索引（按日记本隔离）
        await self._init_tag_indices()

        # 2. 预热日记本名称向量缓存
        self._hydrate_diary_name_cache()

        # 3. 启动文件监视器
        if self.config.enable_watcher:
            self._start_watcher()

        # 4. 构建 Tag 共现矩阵
        await self._build_cooccurrence_matrix()

        # 5. 初始化 EPA 模块（按日记本隔离）
        await self._init_epa_modules()

        # 6. 初始化残差金字塔（按日记本隔离）
        await self._init_residual_pyramids()

        logging.info("[VectorIndex] ✅ System Ready")

        # 标记已初始化
        self._initialized = True

    async def _init_tag_indices(self) -> None:
        """初始化所有日记本的 Tag 索引（懒加载模式）"""
        logging.info("[VectorIndex] 📂 Tag indices will be loaded lazily per diary")
        # Tag 索引采用懒加载策略，在首次使用时才加载

    async def _get_or_load_tag_index(self, diary_name: str) -> Optional[VexusIndex]:
        """
        获取或加载指定日记本的 Tag 索引（懒加载）

        Args:
            diary_name: 日记本名称，如 "反思簇"

        Returns:
            Tag 索引实例，如果该日记本没有标签则返回 None
        """
        # 如果已加载，直接返回
        if diary_name in self.tag_indices:
            return self.tag_indices[diary_name]

        logging.info(f"[VectorIndex] 🔍 Lazy loading tag index for diary: \"{diary_name}\"")

        # 计算文件名：index_tags_{MD5}.usearch
        safe_name = hashlib.md5(diary_name.encode()).hexdigest()
        idx_path = self.config.store_path / f"index_tags_{safe_name}.usearch"

        try:
            if idx_path.exists():
                idx = VexusIndex.load(
                    self.config.dimension,
                    self.config.capacity,
                    str(idx_path)
                )
                self.tag_indices[diary_name] = idx
                logging.info(f"[VectorIndex] ✅ Tag index loaded for \"{diary_name}\"")
                return idx
            else:
                # 创建新索引并从数据库恢复
                idx = VexusIndex(self.config.dimension, self.config.capacity)
                await self._recover_tags_for_diary(idx, diary_name)
                self.tag_indices[diary_name] = idx
                return idx
        except Exception as e:
            logging.error(f"[VectorIndex] ❌ Tag index load failed for \"{diary_name}\": {e}")
            idx = VexusIndex(self.config.dimension, self.config.capacity)
            await self._recover_tags_for_diary(idx, diary_name)
            self.tag_indices[diary_name] = idx
            return idx

    async def _recover_tags_for_diary(self, idx: VexusIndex, diary_name: str) -> None:
        """
        从数据库恢复指定日记本的标签到索引

        Args:
            idx: 索引实例
            diary_name: 日记本名称
        """
        import json
        import struct

        logging.info(f"[VectorIndex] 🔄 Recovering tags for \"{diary_name}\"...")
        try:
            # 使用 Python 手动从数据库读取并添加到索引
            # 因为 Rust 的 recover_from_sqlite 对 tags 表不支持按 diary 过滤
            db = SessionLocal()
            from app.models.database import TagTable, FileTagTable, DiaryFileTable

            tags = db.query(TagTable).join(FileTagTable).join(DiaryFileTable).filter(
                DiaryFileTable.diary_name == diary_name
            ).distinct().all()

            count = 0
            for tag in tags:
                if tag.vector:
                    try:
                        vec = json.loads(tag.vector)
                        vec_bytes = struct.pack(f'{len(vec)}f', *vec)
                        idx.add(tag.id, vec_bytes)
                        count += 1
                    except Exception as e:
                        logging.warning(f"[VectorIndex] ⚠️ Failed to add tag {tag.name}: {e}")

            db.close()
            logging.info(f"[VectorIndex] ✅ Recovered {count} tags for \"{diary_name}\"")
        except Exception as e:
            logging.error(f"[VectorIndex] ❌ Tag recovery failed for \"{diary_name}\": {e}")

    def _start_watcher(self) -> None:
        """启动文件监视器"""
        if self.watcher:
            return

        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        watch_path = self.config.watch_path or str(self.config.data_daily_path)

        class DiaryFileHandler(FileSystemEventHandler):
            def __init__(self, vector_index):
                self.vector_index = vector_index

            def _handle(self, file_path: str, is_delete: bool = False) -> None:
                p = Path(file_path)
                cfg = self.vector_index.config

                # 过滤
                if (p.suffix not in cfg.allowed_extensions or
                    any(p.name.startswith(x) for x in cfg.ignore_prefixes) or
                    any(p.name.endswith(x) for x in cfg.ignore_suffixes) or
                    any(x in cfg.ignore_folders for x in p.parts)):
                    return

                try:
                    rel = p.relative_to(cfg.data_daily_path)
                    char = rel.parts[0] if rel.parts else None
                except ValueError:
                    return

                if not char:
                    return

                # 提取文件名（不包含角色目录）
                file_name = rel.parts[-1] if len(rel.parts) > 1 else str(rel)

                if is_delete:
                    logging.debug(f"[VectorIndex] 🗑️ Deleted: {rel}")
                    asyncio.run_coroutine_threadsafe(
                        self.vector_index._handle_delete(str(rel)),
                        self.vector_index.event_loop
                    )
                else:
                    logging.debug(f"[VectorIndex] 📄 Changed: {rel}")
                    self.vector_index.add_file_to_queue(char, file_name, auto_schedule=False)
                    if len(self.vector_index.pending_files) >= cfg.max_batch_size:
                        asyncio.run_coroutine_threadsafe(
                            self.vector_index._flush_batch(),
                            self.vector_index.event_loop
                        )
                    else:
                        # 调度延迟批处理
                        asyncio.run_coroutine_threadsafe(
                            self.vector_index._schedule_delayed_flush(),
                            self.vector_index.event_loop
                        )

            def on_created(self, e): self._handle(e.src_path) if not e.is_directory else None
            def on_modified(self, e): self._handle(e.src_path) if not e.is_directory else None
            def on_deleted(self, e): self._handle(e.src_path, True) if not e.is_directory else None

        try:
            self.watcher = Observer()
            self.watcher.schedule(DiaryFileHandler(self), watch_path, recursive=True)
            self.watcher.start()
            logging.info(f"[VectorIndex] 👀 File watcher started on: {watch_path}")
        except Exception as e:
            logging.error(f"[VectorIndex] ❌ Failed to start file watcher: {e}")


    # ==================== 1. 懒加载触发 ====================
    async def _get_or_load_diary_index(self, diary_name: str) -> VexusIndex:
        """
        获取或加载指定日记本的索引（懒加载）

        Args:
            diary_name: 日记本名称，如 "反思簇"

        Returns:
            索引实例
        """
        # 如果已加载，直接返回
        if diary_name in self.diary_indices:
            logging.info(f"[VectorIndex] Cache hit for diary: \"{diary_name}\"")
            return self.diary_indices[diary_name]

        logging.info(f"[VectorIndex] 🔍 Lazy loading index for diary: \"{diary_name}\"")

        # 计算文件名：index_diary_{MD5}.usearch
        safe_name = hashlib.md5(diary_name.encode()).hexdigest()
        idx_name = f"diary_{safe_name}"

        # 加载或创建索引
        idx = await self._load_or_build_index(idx_name, "chunks", diary_name)

        # 缓存索引
        self.diary_indices[diary_name] = idx
        return idx

    # ==================== 2. 加载或创建索引 ====================
    async def _load_or_build_index(
        self,
        file_name: str,
        table_type: str,
        filter_diary_name: Optional[str] = None
    ) -> VexusIndex:
        """
        从磁盘加载索引，或创建新索引

        Args:
            file_name: 索引文件名（不含路径和扩展名），如 "diary_5adaf..."
            table_type: 表类型（用于恢复），如 "chunks"
            filter_diary_name: 日记本名称（用于过滤恢复）

        Returns:
            索引实例
        """
        idx_path = self.config.store_path / f"index_{file_name}.usearch"

        try:
            if idx_path.exists():
                idx = VexusIndex.load(
                    self.config.dimension,
                    self.config.capacity,
                    str(idx_path)
                )
            else:
                # 索引文件不存在，需要从数据库恢复
                logging.info(f"[VectorIndex] Index file not found for {file_name}, creating and recovering from DB...")
                idx = VexusIndex(self.config.dimension, self.config.capacity)
                await self._recover_from_database(idx, table_type, filter_diary_name)
        except Exception as e:
            logging.error(f"[VectorIndex] Index load error ({file_name}): {e}")
            logging.warning(f"[VectorIndex] Rebuilding index {file_name} from DB as a fallback...")
            idx = VexusIndex(self.config.dimension, self.config.capacity)
            await self._recover_from_database(idx, table_type, filter_diary_name)

        return idx
    
    async def _recover_from_database(
        self,
        idx: VexusIndex,
        table_type: str,
        diary_name: str
    ) -> None:
        """
        从数据库恢复向量数据到索引（优先使用 Rust，失败则用 Python 手动恢复）

        Args:
            idx: 索引实例
            table_type: 表类型 ("chunks" 或 "tags")
            diary_name: 日记本名称
        """
        import json
        import struct

        logging.info(f"[VectorIndex] 🔄 Recovering {table_type} for \"{diary_name}\" via Rust...")
        try:
            db_path = self.config.store_path / "emotional_companionship.db"
            count = idx.recover_from_sqlite(str(db_path), table_type, diary_name)
            logging.info(f"[VectorIndex] ✅ Recovered {count} vectors via Rust")

            # 如果 Rust 恢复返回 0 条，尝试手动恢复
            if count == 0:
                logging.warning(f"[VectorIndex] ⚠️ Rust recovery returned 0 vectors, trying manual recovery...")
                await self._manual_recover_chunks(idx, diary_name)
                return
        except Exception as e:
            logging.error(f"[VectorIndex] ❌ Rust recovery failed: {e}, trying manual recovery...")
            await self._manual_recover_chunks(idx, diary_name)

    async def _manual_recover_chunks(self, idx: VexusIndex, diary_name: str) -> None:
        """手动从数据库恢复 chunks（Python 实现）"""
        import json
        import struct
        from app.models.database import SessionLocal, ChunkTable, DiaryFileTable

        logging.info(f"[VectorIndex] 🔄 Manual recovering chunks for \"{diary_name}\"...")
        db = SessionLocal()

        try:
            chunks = db.query(ChunkTable).join(DiaryFileTable).filter(
                DiaryFileTable.diary_name == diary_name
            ).all()

            count = 0
            for chunk in chunks:
                if chunk.vector:
                    try:
                        vec = json.loads(chunk.vector)
                        vec_bytes = struct.pack(f'{len(vec)}f', *vec)
                        idx.add(chunk.id, vec_bytes)
                        count += 1
                    except Exception as e:
                        logging.warning(f"[VectorIndex] ⚠️ Failed to add chunk {chunk.id}: {e}")

            logging.info(f"[VectorIndex] ✅ Manual recovered {count} chunks for \"{diary_name}\"")
        except Exception as e:
            logging.error(f"[VectorIndex] ❌ Manual recovery failed: {e}")
        finally:
            db.close()


    # ==================== 核心搜索接口 ====================
    async def search(
        self,
        arg1: Union[str, List[float]],
        arg2: Optional[Union[List[float], int]] = None,
        arg3: Optional[int] = 5,
        arg4: Optional[float] = 0,
    ) -> List[SearchResult]:
        """
        统一搜索接口 - 支持多种参数组合

        参数组合：
        1. search(diary_name, vector, k, tag_boost)
        2. search(vector, k, tag_boost)
        3. search(query_text, k) - 需要先向量化

        Args:
            arg1: 日记本名称
            arg2: 向量 或 k 值
            arg3: k 值 或 tag_boost
            arg4: tag_boost

        Returns:
            搜索结果列表
        """
        try:
            diary_name = None
            query_vec = None
            k = 5
            tag_boost = 0

            # 解析参数
            if isinstance(arg1, str) and isinstance(arg2, list):
                # search(diary_name, vector, k, tag_boost)
                diary_name = arg1
                query_vec = arg2
                k = arg3 if isinstance(arg3, int) else 5
                tag_boost = arg4 if isinstance(arg4, (int, float)) else 0
            elif isinstance(arg1, str):
                # 纯字符串查询，返回空（需要先向量化）
                return []
            elif isinstance(arg1, list):
                # search(vector, k, tag_boost)
                query_vec = arg1
                k = arg2 if isinstance(arg2, int) else 5
                tag_boost = arg3 if isinstance(arg3, (int, float)) else 0

            if not query_vec:
                return []

            return await self._search_specific_index(diary_name, query_vec, k, tag_boost)

        except Exception as e:
            logging.error(f"[VectorIndex] Search error: {e}")
            return []

    async def _search_specific_index(
        self,
        diary_name: str,
        vector: List[float],
        k: int,
        tag_boost: float,
    ) -> List[SearchResult]:
        """
        在指定日记本索引中搜索

        Args:
            diary_name: 日记本名称
            vector: 查询向量
            k: 返回结果数量
            tag_boost: 标签增强因子

        Returns:
            搜索结果列表
        """
        idx = await self._get_or_load_diary_index(diary_name)

        # 检查索引是否有数据
        try:
            stats = idx.stats()
            if stats.total_vectors == 0:
                return []
        except Exception:
            pass

        # 应用 Tag 增强（如果启用）
        search_vec = vector
        tag_info = None

        if tag_boost > 0:
            logging.info(f"[VectorIndex] ✅ TagBoost 已启用，开始增强向量 (diary={diary_name})...")
            boost_result = await self.apply_tag_boost(vector, tag_boost, diary_name)
            search_vec = boost_result.vector
            tag_info = boost_result.info
        else:
            logging.info(f"[VectorIndex] ❌ TagBoost 未启用，使用原始向量")

        # 维度检查
        if len(search_vec) != self.config.dimension:
            logging.error(f"[VectorIndex] Dimension mismatch! Expected {self.config.dimension}, got {len(search_vec)}")
            return []

        # 转换为 bytes
        try:
            search_buffer = self._serialize_vector(search_vec)
        except Exception as err:
            logging.error(f"[VectorIndex] Buffer conversion failed: {err}")
            return []

        # 执行搜索
        try:
            results = idx.search(search_buffer, k)
        except Exception as e:
            logging.error(f"[VectorIndex] Vexus search failed for \"{diary_name}\": {e}")
            return []

        # 从数据库获取完整内容（批量查询优化）
        db: Session = SessionLocal()
        try:
            # 批量查询所有 chunks
            chunk_ids = [res.id for res in results]
            chunks = db.query(ChunkTable).filter(ChunkTable.id.in_(chunk_ids)).all()
            chunk_map = {chunk.id: chunk for chunk in chunks}

            # 构建结果（保持原始顺序）
            search_results = []
            for res in results:
                chunk = chunk_map.get(res.id)
                if chunk and chunk.file:
                    search_results.append(SearchResult(
                        text=chunk.content,
                        score=res.score,
                        source_file=Path(chunk.file.path).name,
                        full_path=chunk.file.path,
                        updated_at=chunk.file.updated_at,
                        matched_tags=tag_info.get('matched_tags', []) if tag_info else [],
                        boost_factor=tag_info.get('boost_factor', 0) if tag_info else 0,
                    ))
            return search_results
        finally:
            db.close()

    # ==================== TagMemo 核心增强算法 ====================
    async def _apply_tag_memo_v3(
        self,
        vector: Union[List[float], np.ndarray],
        base_tag_boost: float,
        diary_name: Optional[str] = None,
    ) -> TagBoostResult:
        """
        EPA、残差金字塔、共现矩阵等多种技术进行智能语义增强
        Args:
            vector: 原始查询向量
            base_tag_boost: 基础增强因子 (0-1)
            diary_name: 日记本名称（用于获取对应的 EPA 和残差金字塔模块）

        Returns:
            TagBoostResult 包含增强后的向量和调试信息
        """
        original_float32 = self._ensure_float32(vector)
        dim = len(original_float32)

        try:
            # 获取配置
            config = self.rag_params.get('VectorIndex', {})

            # 获取或初始化对应日记本的 EPA 和残差金字塔模块
            epa = await self._get_or_init_epa_module(diary_name) if diary_name else None
            residual_pyramid = await self._get_or_init_residual_pyramid(diary_name) if diary_name else None

            # Step 1: EPA 分析
            if epa and epa.initialized:
                epa_result = epa.project(original_float32)
                resonance = epa.detect_cross_domain_resonance(original_float32)
            else:
                # Fallback: 使用默认空结果
                epa_result = {
                    "projections": None,
                    "probabilities": None,
                    "entropy": 1.0,
                    "logic_depth": 0.0,
                    "dominant_axes": []
                }
                resonance = {"resonance": 0.0, "bridges": []}

            query_world = (
                epa_result['dominant_axes'][0]['label']
                if epa_result['dominant_axes']
                else 'Unknown'
            )

            logging.info("当前所处的世界：" + query_world)

            # Step 2: 残差金字塔分析
            if residual_pyramid:
                pyramid = residual_pyramid.analyze(original_float32)
                features = pyramid['features']
            else:
                # Fallback: 使用默认空结果
                features = {
                    "depth": 0,
                    "coverage": 0.0,
                    "novelty": 1.0,
                    "coherence": 0.0,
                    "tag_memo_activation": 0.0,
                    "expansion_signal": 1.0,
                }

            # Step 3: 动态调整策略
            logic_depth = epa_result['logic_depth']
            entropy_penalty = epa_result['entropy']
            resonance_boost = np.log1p(resonance['resonance'])

            activation_multiplier = 0.5 + features['tag_memo_activation'] * 1.5
            dynamic_boost_factor = (
                logic_depth * (1 + resonance_boost) / (1 + entropy_penalty * 0.5)
            ) * activation_multiplier

            effective_tag_boost = base_tag_boost * min(2.0, max(0.3, dynamic_boost_factor))

            # Step 4: 收集金字塔 Tags 并应用世界观门控
            all_tags = []
            seen_tag_ids = set()

            levels = pyramid.get('levels', [])
            for level in levels:
                tags = level.get('tags', [])
                for t in tags:
                    if not t or t.get('id') in seen_tag_ids:
                        continue

                    tag_name = t.get('name', '').lower() if t.get('name') else ''

                    # 语言置信度补偿
                    lang_penalty = self._apply_language_compensation(
                        tag_name, query_world, config
                    )

                    # 世界观门控
                    layer_decay = 0.7 ** level.get('level', 0)
                    # 个体相关度微调：如果核心标签本身与查询高度相关，给予额外奖励 (0.95 ~ 1.05x)
                    all_tags.append({
                        **t,
                        'adjustedWeight': (
                            t.get('contribution', t.get('weight', 0))
                            * layer_decay
                            * lang_penalty
                        ),
                    })
                    seen_tag_ids.add(t['id'])

            # Step 6: 逻辑分支拉回（Tag 共现矩阵）
            if all_tags and self.tag_cooccurrence_matrix:
                top_tags = sorted(
                    all_tags, key=lambda x: x['adjustedWeight'], reverse=True
                )[:5]

                for parent_tag in top_tags:
                    related = self.tag_cooccurrence_matrix.get(parent_tag['id'], {})
                    sorted_related = sorted(
                        related.items(), key=lambda x: x[1], reverse=True
                    )[:4]

                    for rel_id, _ in sorted_related:
                        if rel_id not in seen_tag_ids:
                            all_tags.append({
                                'id': rel_id,
                                'adjustedWeight': parent_tag['adjustedWeight'] * 0.5,
                                'isPullback': True
                            })
                            seen_tag_ids.add(rel_id)

            # Step 5: 批量获取向量
            all_tag_ids = [t['id'] for t in all_tags]
            db: Session = SessionLocal()
            try:
                rows = db.query(TagTable).filter(TagTable.id.in_(all_tag_ids)).all()
                tag_data_map = {
                    row.id: {'id': row.id, 'name': row.name, 'vector': row.vector}
                    for row in rows
                }
            finally:
                db.close()

            # Step 6: 构建上下文向量并融合
            context_vec = np.zeros(dim, dtype=np.float32)
            total_weight = 0

            for t in all_tags:
                data = tag_data_map.get(t['id'])
                if data and data.get('vector'):
                    vec_data = data['vector']
                    # 处理可能的多种格式：JSON 字符串、bytes 或 list
                    if isinstance(vec_data, bytes):
                        vec_data = vec_data.decode('utf-8')
                    if isinstance(vec_data, str):
                        vec_data = json.loads(vec_data)
                    v = np.array(vec_data, dtype=np.float32)
                    context_vec += v * t['adjustedWeight']
                    total_weight += t['adjustedWeight']

            if total_weight > 0:
                context_vec /= total_weight
                norm = np.linalg.norm(context_vec)
                if norm > 1e-9:
                    context_vec /= norm

            # 融合
            fused = (
                (1 - effective_tag_boost) * original_float32 +
                effective_tag_boost * context_vec
            )
            norm = np.linalg.norm(fused)
            if norm > 1e-9:
                fused /= norm

            # 构建结果信息
            return TagBoostResult(
                vector=fused.tolist(),
                info={
                    'matchedTags': effective_tag_boost,
                    'boostFactor': float(effective_tag_boost),
                    'epa': {
                        'logicDepth': float(logic_depth),
                        'entropy': float(entropy_penalty),
                        'resonance': float(resonance['resonance'])
                    },
                    'pyramid': features
                }
            )

        except Exception as e:
            logging.error(f"[VectorIndex] TagMemo V3 error: {e}, falling back to simple boost")
            return TagBoostResult(vector=original_float32.tolist(), info=None)

    def _ensure_float32(self, vector: Union[List[float], np.ndarray]) -> np.ndarray:
        """确保向量是 float32 类型"""
        if isinstance(vector, list):
            return np.array(vector, dtype=np.float32)
        elif vector.dtype != np.float32:
            return vector.astype(np.float32)
        return vector

    async def apply_tag_boost(
        self,
        vector: List[float],
        tag_boost: float,
        diary_name: Optional[str] = None,
    ) -> TagBoostResult:
        """
        公共接口：应用 TagMemo V3 增强算法

        Args:
            vector: 原始向量
            tag_boost: 增强因子 (0-1)
            diary_name: 日记本名称（用于获取对应的 tag 索引和模块）

        Returns:
            TagBoostResult 包含增强后的向量和调试信息
        """

        return await self._apply_tag_memo_v3(vector, tag_boost, diary_name)

    def get_epa_analysis(self, vector: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        获取向量的 EPA 分析数据（逻辑深度、共振等）

        Args:
            vector: 输入向量

        Returns:
            包含 logic_depth, entropy, resonance, dominant_axes 的字典
        """
        if not self.epa or not self.epa.initialized:
            return {
                "logic_depth": 0.5,
                "resonance": 0.0,
                "entropy": 0.5,
                "dominant_axes": []
            }

        # 确保是 float32 numpy 数组
        vec = self._ensure_float32(vector)

        # 执行投影和共振检测
        projection = self.epa.project(vec)
        resonance = self.epa.detect_cross_domain_resonance(vec)

        return {
            "logic_depth": projection.get("logic_depth", 0.5),
            "entropy": projection.get("entropy", 0.5),
            "resonance": resonance.get("resonance", 0.0),
            "dominant_axes": projection.get("dominant_axes", [])
        }

    # ==================== 3. 添加向量 ====================
    async def add_vector(self, diary_name: str, id: int, vector_buffer: bytes) -> None:
        """
        向日记本索引添加向量

        Args:
            diary_name: 日记本名称
            id: 向量ID（如chunk_id）
            vector_buffer: 向量数据的bytes
        """
        logging.info(f"[VectorIndex] ➕ Adding vector {id} to diary \"{diary_name}\"")

        # 通过懒加载获取索引
        idx = await self._get_or_load_diary_index(diary_name)

        # 添加向量到索引
        try:
            idx.add(id, vector_buffer)
        except Exception as e:
            error_msg = str(e)
            if "Duplicate" in error_msg:
                # 处理重复ID：先删除再添加（upsert）
                logging.warning(f"[VectorIndex] ⚠️ Duplicate ID {id}, performing upsert...")
                if hasattr(idx, "remove"):
                    idx.remove(id)
                idx.add(id, vector_buffer)
            else:
                raise

        # 安排延迟保存
        self._schedule_index_save(diary_name)

    async def add_vectors(self, diary_name: str, vectors: List[Tuple[int, bytes]]) -> None:
        """
        批量添加向量

        Args:
            diary_name: 日记本名称
            vectors: 向量数组，每个元素为 (id, vec) 元组
        """
        logging.info(f"[VectorIndex] Adding {len(vectors)} vectors to diary \"{diary_name}\"")

        idx = await self._get_or_load_diary_index(diary_name)

        for id, vec in vectors:
            try:
                idx.add(id, vec)
            except Exception as e:
                error_msg = str(e)
                if "Duplicate" in error_msg:
                    if hasattr(idx, "remove"):
                        idx.remove(id)
                    idx.add(id, vec)

        self._schedule_index_save(diary_name)

    # ==================== 4. 保存到磁盘 ====================
    def _schedule_index_save(self, diary_name: str) -> None:
        """
        安排延迟保存（防抖）

        Args:
            diary_name: 日记本名称
        """
        # 如果已有任务，取消它
        if diary_name in self.save_tasks and self.save_tasks[diary_name] is not None:
            self.save_tasks[diary_name].cancel()

        # 创建新的延迟保存任务
        async def save_task():
            await asyncio.sleep(self.config.index_save_delay)
            await self._save_index_to_disk(diary_name)
            self.save_tasks[diary_name] = None

        task = asyncio.create_task(save_task())
        self.save_tasks[diary_name] = task

    async def _save_index_to_disk(self, diary_name: str, is_tag_index: bool = False) -> None:
        """
        立即保存索引到磁盘

        Args:
            diary_name: 日记本名称
            is_tag_index: 是否为 tag 索引
        """
        try:
            if is_tag_index:
                # 保存标签索引
                safe_name = hashlib.md5(diary_name.encode()).hexdigest()
                idx = self.tag_indices.get(diary_name)
                if idx is None:
                    logging.warning(f"[VectorIndex] ⚠️ No tag index found for \"{diary_name}\", skipping save.")
                    return
                file_path = self.config.store_path / f"index_tags_{safe_name}.usearch"
                idx.save(str(file_path))
                logging.info(f"[VectorIndex] 💾 Saved tag index: {diary_name}")
            else:
                # 保存日记索引
                safe_name = hashlib.md5(diary_name.encode()).hexdigest()
                idx = self.diary_indices.get(diary_name)
                if idx is None:
                    logging.warning(f"[VectorIndex] ⚠️ No index found for \"{diary_name}\", skipping save.")
                    return
                file_path = self.config.store_path / f"index_diary_{safe_name}.usearch"
                idx.save(str(file_path))
            logging.info(f"[VectorIndex] 💾 Saved index: {diary_name}")
        except Exception as e:
            logging.error(f"[VectorIndex] ❌ Save failed for {diary_name}: {e}")

    # ==================== 向量缓存系统 ====================
    def _hydrate_diary_name_cache(self) -> None:
        """从数据库预热日记本名称向量缓存"""
        logging.info("[VectorIndex] 🔄 Hydrating diary name vectors...")
        db: Session = SessionLocal()
        try:
            kv_entries = db.query(KVStoreTable).filter(
                KVStoreTable.key.like("diary_name:%")
            ).all()
            count = 0
            for entry in kv_entries:
                if entry.vector:
                    try:
                        name = entry.key.split(":", 1)[1]
                        vector = json.loads(entry.vector)
                        self.diary_name_vector_cache[name] = vector
                        count += 1
                    except (json.JSONDecodeError, IndexError, TypeError):
                        pass
            logging.info(f"[VectorIndex] ✅ Hydrated {count} diary name vectors")
        finally:
            db.close()

    async def _fetch_and_cache_diary_name_vector(self, name: str) -> Optional[List[float]]:
        """获取并缓存日记本名称向量"""
        try:
            async with EmbeddingService() as embedding_service:
                vectors = await embedding_service.get_embeddings_batch([name])
                if vectors and vectors[0]:
                    self.diary_name_vector_cache[name] = vectors[0]
                    # 保存到数据库
                    self._save_kv_store(f"diary_name:{name}", vectors[0])
                    return vectors[0]
        except Exception as e:
            logging.error(f"[VectorIndex] Failed to vectorize diary name {name}: {e}")
        return None

    def _save_kv_store(self, key: str, vector: List[float], value: Optional[str] = None) -> None:
        """保存向量到 KV Store"""
        db: Session = SessionLocal()
        try:
            entry = db.query(KVStoreTable).filter(KVStoreTable.key == key).first()
            vector_json = json.dumps(vector)
            now = int(time.time())

            if entry:
                entry.vector = vector_json
                entry.value = value
                entry.updated_at = now
            else:
                new_entry = KVStoreTable(
                    key=key,
                    value=value,
                    vector=vector_json,
                    updated_at=now
                )
                db.add(new_entry)
            db.commit()
        except Exception as e:
            db.rollback()
            logging.error(f"[VectorIndex] Failed to save KV store {key}: {e}")
        finally:
            db.close()

    # ==================== 标签提取 ====================
    def extract_tags(self, content: str) -> List[str]:
        """
        从内容中提取标签

        Args:
            content: 文本内容

        Returns:
            标签列表
        """
        # 匹配 Tag: 行
        tag_lines = re.findall(r'Tag:\s*(.+)$', content, re.MULTILINE | re.IGNORECASE)
        if not tag_lines:
            return []

        # 分割标签
        all_tags = []
        for line in tag_lines:
            split_tags = re.split(r'[,，、;|｜]', line)
            all_tags.extend(t.strip() for t in split_tags if t.strip())

        # 清理标签
        tags = []
        for tag in all_tags:
            cleaned = re.sub(r'[。.]+$', '', tag).strip()
            cleaned = self._prepare_text_for_embedding(cleaned)
            if cleaned and cleaned != '[EMPTY_CONTENT]':
                tags.append(cleaned)

        # 应用黑名单
        if self.config.tag_blacklist_super:
            super_regex = re.compile('|'.join(map(re.escape, self.config.tag_blacklist_super)))
            tags = [super_regex.sub('', t).strip() for t in tags]

        tags = [t for t in tags if t and t not in self.config.tag_blacklist]

        return list(set(tags))  # 去重

    def _prepare_text_for_embedding(self, text: str) -> str:
        """
        预处理文本用于嵌入

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        cleaned = text
        # 清理空白字符
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r' *\n *', '\n', cleaned)
        cleaned = re.sub(r'\n{2,}', '\n', cleaned)
        cleaned = cleaned.strip()
        return cleaned if cleaned else '[EMPTY_CONTENT]'

    # ==================== 批处理支持 ====================
    def add_file_to_queue(self, character_name: str, file_path: str, auto_schedule: bool = True) -> None:
        """
        添加文件到批处理队列

        Args:
            character_name: 角色名称
            file_path: 文件路径（相对于 daily/ 目录）
            auto_schedule: 是否自动调度批处理（监视器线程设为 False）
        """
        key = f"{character_name}:{file_path}"
        self.pending_files.add(key)
        if auto_schedule:
            try:
                self._schedule_batch_flush()
            except RuntimeError:
                # 事件循环未运行，静默忽略（由调用方手动触发）
                pass
        logging.debug(f"[VectorIndex] 📥 Added file to queue: {key} (queue size: {len(self.pending_files)})")

    def _schedule_batch_flush(self) -> None:
        """调度批处理（带延迟）"""
        if self.batch_timer and not self.batch_timer.done():
            self.batch_timer.cancel()

        self.batch_timer = asyncio.create_task(self._schedule_delayed_flush())

    async def _schedule_delayed_flush(self) -> None:
        """延迟批处理（供监视器线程调用）"""
        await asyncio.sleep(self.config.batch_delay)
        await self._flush_batch()

    async def _flush_batch(self) -> None:
        """
        批量处理文件队列

        流程：
        1. 从队列中取出一批文件
        2. 检查文件是否需要更新
        3. 批量向量化
        4. 批量写入数据库
        5. 批量更新向量索引
        """
        if self.is_processing or not self.pending_files:
            return

        self.is_processing = True
        batch_keys = list(self.pending_files)[:self.config.max_batch_size]

        # 取消定时器
        if self.batch_timer and not self.batch_timer.done():
            self.batch_timer.cancel()
            self.batch_timer = None

        logging.info(f"[VectorIndex] 🚌 Processing batch of {len(batch_keys)} files...")

        try:
            # 统计信息
            skipped_files = []  # 未变化的文件
            processed_files = []  # 需要处理的文件

            # 解析文件列表
            files_to_process = []
            db: Session = SessionLocal()
            try:
                for key in batch_keys:
                    character_name, file_path = key.split(":", 1)
                    diary_dir = self._get_diary_dir(character_name)
                    full_path = diary_dir / file_path

                    if not full_path.exists():
                        logging.warning(f"[VectorIndex] ❌ File not found: {full_path}")
                        self.pending_files.discard(key)
                        self.file_retry_count.pop(key, None)
                        continue

                    # 获取文件元数据
                    try:
                        content = full_path.read_text(encoding='utf-8')
                        checksum = self._calculate_checksum(content)
                        mtime = int(full_path.stat().st_mtime)
                        size = len(content.encode('utf-8'))

                        # 检查是否需要更新
                        existing = db.query(DiaryFileTable).filter(
                            DiaryFileTable.path == file_path
                        ).first()

                        # 检查索引文件是否存在
                        safe_name = hashlib.md5(character_name.encode()).hexdigest()
                        idx_path = self.config.store_path / f"index_diary_{safe_name}.usearch"
                        index_exists = idx_path.exists()

                        # 获取角色信息 (在判断前获取)
                        character_service = CharacterService()
                        character = character_service.get_character_by_name(character_name)
                        if not character:
                            logging.warning(f"[VectorIndex] ❌ Character not found: {character_name}")
                            self.pending_files.discard(key)
                            self.file_retry_count.pop(key, None)
                            continue

                        if existing and existing.checksum == checksum and index_exists:
                            # 文件未变化且索引存在，跳过
                            self.pending_files.discard(key)
                            self.file_retry_count.pop(key, None)
                            skipped_files.append({
                                "character_name": character_name,
                                "file_path": file_path
                            })
                            continue
                        elif existing and existing.checksum == checksum and not index_exists:
                            # 文件未变化但索引不存在，需要重建索引
                            logging.info(f"[VectorIndex] 🔄 Index missing for {file_path}, forcing rebuild...")
                            # 强制加入处理队列进行索引重建

                        # 加入待处理列表的条件：
                        # 1. 文件不存在
                        # 2. 文件已变化 (checksum 不同)
                        # 3. 索引文件不存在 (即使 checksum 相同，也需要重建)
                        if not existing or existing.checksum != checksum or not index_exists:
                            files_to_process.append({
                            "key": key,
                            "character_name": character_name,
                            "file_path": file_path,
                            "diary_name": character.name,
                            "content": content,
                            "checksum": checksum,
                            "mtime": mtime,
                            "size": size,
                            "existing_id": existing.id if existing else None
                        })
                        processed_files.append({
                            "character_name": character_name,
                            "file_path": file_path,
                            "diary_name": character.name
                        })
                    except Exception as e:
                        logging.warning(f"[VectorIndex] ❌ Error reading file {full_path}: {e}")
            finally:
                db.close()

            # 打印统计日志
            logging.info(f"[VectorIndex] 📊 Batch Statistics:")
            logging.info(f"[VectorIndex]   ⏭️  Skipped (unchanged): {len(skipped_files)} files")
            for f in skipped_files:
                logging.info(f"[VectorIndex]      - {f['character_name']}: {f['file_path']}")
            logging.info(f"[VectorIndex]   🔄 To Process: {len(processed_files)} files")
            for f in processed_files:
                logging.info(f"[VectorIndex]      - {f['diary_name']}: {f['file_path']}")

            if not files_to_process:
                logging.info("[VectorIndex] ✅ No files need processing (all skipped)")
                return

            # 批量处理：分块 + 提取标签
            all_chunks_with_meta = []
            unique_tags = set()

            for file_info in files_to_process:
                chunks = chunk_text(file_info["content"])
                valid_chunks = [self._prepare_text_for_embedding(c) for c in chunks]
                valid_chunks = [c for c in valid_chunks if c != '[EMPTY_CONTENT]']

                tags = self.extract_tags(file_info["content"])
                unique_tags.update(tags)

                for i, chunk_content in enumerate(valid_chunks):
                    all_chunks_with_meta.append({
                        "text": chunk_content,
                        "diary_name": file_info["diary_name"],
                        "file_info": file_info,
                        "chunk_index": i
                    })

            if not all_chunks_with_meta:
                logging.warning("[VectorIndex] No valid chunks found")
                return

            # 批量向量化 chunks
            chunk_texts = [item["text"] for item in all_chunks_with_meta]
            async with EmbeddingService() as embedding_service:
                chunk_vectors = await embedding_service.get_embeddings_batch(chunk_texts)

            # 批量向量化新标签
            tag_cache = {}
            new_tags = set()
            db: Session = SessionLocal()
            try:
                for tag in unique_tags:
                    existing = db.query(TagTable).filter(TagTable.name == tag).first()
                    if existing and existing.vector:
                        tag_cache[tag] = {"id": existing.id, "vector": json.loads(existing.vector)}
                    else:
                        cleaned = self._prepare_text_for_embedding(tag)
                        if cleaned != '[EMPTY_CONTENT]':
                            new_tags.add(cleaned)

                if new_tags:
                    async with EmbeddingService() as embedding_service:
                        tag_vectors_list = await embedding_service.get_embeddings_batch(list(new_tags))
                        tag_vector_map = dict(zip(new_tags, tag_vectors_list))

                        # 插入新标签
                        for tag, vector in tag_vector_map.items():
                            if vector:
                                vec_bytes = self._serialize_vector(vector)
                                new_tag = TagTable(name=tag, vector=json.dumps(vector))
                                db.add(new_tag)
                                db.flush()
                                tag_cache[tag] = {"id": new_tag.id, "vector": vector}
            finally:
                db.close()

            # 关联向量到元数据
            for i, item in enumerate(all_chunks_with_meta):
                if i < len(chunk_vectors) and chunk_vectors[i]:
                    item["vector"] = chunk_vectors[i]

            # 按日记本分组
            updates_by_diary: Dict[str, List[Dict]] = {}
            deletions_by_diary: Dict[str, List[int]] = {}

            db: Session = SessionLocal()
            try:
                updated_at = int(time.time())

                for file_info in files_to_process:
                    diary_name = file_info["diary_name"]
                    file_path = file_info["file_path"]

                    if diary_name not in updates_by_diary:
                        updates_by_diary[diary_name] = []

                    # 获取或创建文件记录
                    if file_info["existing_id"]:
                        file_id = file_info["existing_id"]
                        # 删除旧 chunks
                        old_chunks = db.query(ChunkTable).filter(ChunkTable.file_id == file_id).all()
                        if old_chunks:
                            if diary_name not in deletions_by_diary:
                                deletions_by_diary[diary_name] = []
                            deletions_by_diary[diary_name].extend([c.id for c in old_chunks])
                        db.query(ChunkTable).filter(ChunkTable.file_id == file_id).delete()
                        db.query(FileTagTable).filter(FileTagTable.file_id == file_id).delete()

                        existing_file = db.query(DiaryFileTable).get(file_id)
                        existing_file.checksum = file_info["checksum"]
                        existing_file.mtime = file_info["mtime"]
                        existing_file.size = file_info["size"]
                        existing_file.updated_at = updated_at
                    else:
                        new_file = DiaryFileTable(
                            path=file_path,
                            diary_name=diary_name,
                            checksum=file_info["checksum"],
                            mtime=file_info["mtime"],
                            size=file_info["size"],
                            updated_at=updated_at
                        )
                        db.add(new_file)
                        db.flush()
                        file_id = new_file.id

                    # 插入新 chunks
                    file_chunks = [item for item in all_chunks_with_meta
                                   if item["file_info"]["file_path"] == file_path]

                    for chunk_item in file_chunks:
                        if "vector" in chunk_item and chunk_item["vector"]:
                            vec_bytes = self._serialize_vector(chunk_item["vector"])
                            chunk_entry = ChunkTable(
                                file_id=file_id,
                                chunk_index=chunk_item["chunk_index"],
                                content=chunk_item["text"],
                                vector=json.dumps(chunk_item["vector"])
                            )
                            db.add(chunk_entry)
                            db.flush()

                            updates_by_diary[diary_name].append({
                                "id": chunk_entry.id,
                                "vec": vec_bytes
                            })

                    # 处理标签关联
                    tags = self.extract_tags(file_info["content"])
                    for tag in tags:
                        if tag in tag_cache:
                            existing_rel = db.query(FileTagTable).filter(
                                FileTagTable.file_id == file_id,
                                FileTagTable.tag_id == tag_cache[tag]["id"]
                            ).first()
                            if not existing_rel:
                                db.add(FileTagTable(file_id=file_id, tag_id=tag_cache[tag]["id"]))

                db.commit()
            except Exception as e:
                db.rollback()
                logging.error(f"[VectorIndex] Database error: {e}")
                raise
            finally:
                db.close()

            # 处理删除（从向量索引中移除）
            for diary_name, chunk_ids in deletions_by_diary.items():
                idx = await self._get_or_load_diary_index(diary_name)
                if idx and hasattr(idx, "remove"):
                    for chunk_id in chunk_ids:
                        try:
                            idx.remove(chunk_id)
                        except Exception as e:
                            logging.warning(f"[VectorIndex] Failed to remove chunk {chunk_id}: {e}")

            # 添加新向量到索引
            for diary_name, chunks in updates_by_diary.items():
                idx = await self._get_or_load_diary_index(diary_name)
                for chunk in chunks:
                    try:
                        idx.add(chunk["id"], chunk["vec"])
                    except Exception as e:
                        if "Duplicate" in str(e):
                            try:
                                if hasattr(idx, "remove"):
                                    idx.remove(chunk["id"])
                                idx.add(chunk["id"], chunk["vec"])
                            except Exception as retry_err:
                                logging.error(f"[VectorIndex] Failed to upsert {chunk['id']}: {retry_err}")

                self._schedule_index_save(diary_name)

            # 处理 Tag 索引更新（按日记本分组）
            tag_updates_by_diary: Dict[str, Dict] = {}
            for file_info in files_to_process:
                diary_name = file_info["diary_name"]
                if diary_name not in tag_updates_by_diary:
                    tag_updates_by_diary[diary_name] = {}
                # 只将当前文件相关的 tag 添加到对应日记本
                tags = self.extract_tags(file_info["content"])
                for tag in tags:
                    if tag in tag_cache:
                        tag_data = tag_cache[tag]
                        tag_updates_by_diary[diary_name][tag] = tag_data

            for diary_name, diary_tag_cache in tag_updates_by_diary.items():
                if diary_tag_cache:
                    try:
                        tag_index = await self._get_or_load_tag_index(diary_name)
                        if tag_index:
                            logging.info(f"[VectorIndex] 🏷️ Building tag index for \"{diary_name}\" with {len(diary_tag_cache)} tags...")
                            for tag, tag_data in diary_tag_cache.items():
                                if tag_data and "vector" in tag_data:
                                    vector_bytes = self._serialize_vector(tag_data["vector"])
                                    try:
                                        tag_index.add(tag_data["id"], vector_bytes)
                                    except Exception as e:
                                        if "Duplicate" in str(e):
                                            if hasattr(tag_index, "remove"):
                                                tag_index.remove(tag_data["id"])
                                            tag_index.add(tag_data["id"], vector_bytes)
                            # 安排保存 Tag 索引
                            self._schedule_tag_index_save(diary_name)
                            logging.info(f"[VectorIndex] ✅ Tag index updated for \"{diary_name}\" with {len(diary_tag_cache)} tags")
                    except Exception as e:
                        logging.error(f"[VectorIndex] ❌ Failed to update tag index for \"{diary_name}\": {e}")

            # 清理已处理的文件
            for key in batch_keys:
                if key in self.pending_files:
                    self.pending_files.discard(key)
                self.file_retry_count.pop(key, None)

            logging.info(f"[VectorIndex] ✅ Batch complete. Updated {len(updates_by_diary)} diary indices.")

            # 优化：数据更新后，异步重建共现矩阵
            asyncio.create_task(self._build_cooccurrence_matrix())

        except Exception as e:
            logging.error(f"[VectorIndex] ❌ Batch processing failed: {e}")

            # 错误重试机制
            for key in batch_keys:
                retry_count = self.file_retry_count.get(key, 0) + 1
                if retry_count >= self.config.max_file_retries:
                    logging.error(f"[VectorIndex] ⛔ File {key} failed {retry_count} times. Removing permanently.")
                    self.pending_files.discard(key)
                    self.file_retry_count.pop(key, None)
                else:
                    self.file_retry_count[key] = retry_count
                    logging.warning(f"[VectorIndex] ⚠️ File {key} retry {retry_count}/{self.config.max_file_retries}")

        finally:
            self.is_processing = False
            # 继续处理剩余文件
            if self.pending_files:
                asyncio.create_task(self._flush_batch())

    async def sync_character_diaries(
        self,
        name: str
    ) -> Dict[str, Any]:
        """
        同步指定角色的所有日记文件（使用批处理队列）

        Args:
            name: 日记名称

        Returns:
            同步结果：{queued, total}
        """
        diary_dir = self._get_diary_dir(name)

        if not diary_dir.exists():
            logging.warning(f"[VectorIndex] Diary directory not found: {diary_dir}")
            return {"queued": 0, "total": 0}

        # 获取所有.txt文件
        txt_files = sorted(diary_dir.glob("*.txt"))
        logging.info(f"[VectorIndex] Found {len(txt_files)} diary files for {name}")

        # 将所有文件添加到批处理队列
        for file_path in txt_files:
            relative_path = file_path.name  # 只保留文件名
            self.add_file_to_queue(name, relative_path)

        logging.info(f"[VectorIndex] ✅ Added {len(txt_files)} files to batch queue for {name}")

        # 手动触发批处理（确保标签被提取和存储）
        if self.pending_files and self.event_loop:
            try:
                asyncio.create_task(self._flush_batch())
            except RuntimeError:
                # 如果没有运行中的事件循环，使用同步调度
                self._schedule_batch_flush()

        return {"queued": len(txt_files), "total": len(txt_files)}

    # ==================== 辅助方法 ====================

    def _serialize_vector(self, vector: List[float]) -> bytes:
        """
        将向量列表序列化为 bytes 格式

        Args:
            vector: 向量列表

        Returns:
            序列化后的bytes
        """
        return struct.pack(f'{len(vector)}f', *vector)

    async def _process_file_tags(
        self,
        db: Session,
        file_id: int,
        tags: List[str],
        diary_name: str,
    ) -> None:
        """
        处理文件标签：保存标签到数据库并更新索引

        Args:
            db: 数据库会话
            file_id: 文件ID
            tags: 标签列表
        """
        if not tags:
            return

        # 删除旧的文件-标签关联
        db.query(FileTagTable).filter(FileTagTable.file_id == file_id).delete()

        # 获取或创建标签
        tag_cache = {}
        for tag in tags:
            existing_tag = db.query(TagTable).filter(TagTable.name == tag).first()
            if existing_tag:
                tag_cache[tag] = existing_tag.id
            else:
                # 新标签，需要向量化
                tag_cache[tag] = None  # 标记为需要创建

        # 批量向量化新标签
        new_tags = [t for t, tid in tag_cache.items() if tid is None]
        if new_tags:
            try:
                async with EmbeddingService() as embedding_service:
                    vectors = await embedding_service.get_embeddings_batch(new_tags)

                for tag, vector in zip(new_tags, vectors):
                    if vector:
                        new_tag = TagTable(
                            name=tag,
                            vector=json.dumps(vector)
                        )
                        db.add(new_tag)
                        db.flush()
                        tag_cache[tag] = new_tag.id

                        # 添加到对应日记本的 Tag 索引
                        tag_index = await self._get_or_load_tag_index(diary_name)
                        if tag_index:
                            vector_bytes = self._serialize_vector(vector)
                            try:
                                tag_index.add(new_tag.id, vector_bytes)
                            except Exception as e:
                                if "Duplicate" in str(e):
                                    if hasattr(tag_index, "remove"):
                                        tag_index.remove(new_tag.id)
                                    tag_index.add(new_tag.id, vector_bytes)
            except Exception as e:
                logging.error(f"[VectorIndex] Failed to vectorize tags: {e}")

        # 创建文件-标签关联
        for tag, tag_id in tag_cache.items():
            if tag_id is not None:
                association = FileTagTable(file_id=file_id, tag_id=tag_id)
                db.add(association)

        # 安排保存 Tag 索引
        self._schedule_tag_index_save(diary_name)

    def _schedule_tag_index_save(self, diary_name: str) -> None:
        """安排指定日记本的 Tag 索引延迟保存"""
        if diary_name in self.tag_save_tasks and self.tag_save_tasks[diary_name] is not None:
            self.tag_save_tasks[diary_name].cancel()

        async def save_task():
            await asyncio.sleep(self.config.tag_index_save_delay)
            await self._save_index_to_disk(diary_name, is_tag_index=True)
            self.tag_save_tasks[diary_name] = None

        task = asyncio.create_task(save_task())
        self.tag_save_tasks[diary_name] = task

    def _get_diary_dir(self, name: str) -> Path:
        """
        获取角色日记目录路径

        Args:
            name: 角色名称

        Returns:
            日记目录路径 (data/daily/{sanitized_name}/)
        """
        # Sanitize name the same way CharacterService does
        import re
        sanitized = re.sub(r'[\\/:*?"<>|]', '', name.strip())
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = sanitized[:100] if len(sanitized) > 100 else sanitized
        sanitized = sanitized or 'unnamed'

        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        return project_root / "data" / "daily" / sanitized

    def _calculate_checksum(self, content: str) -> str:
        """
        计算文本内容的MD5校验和

        Args:
            content: 文本内容

        Returns:
            MD5哈希字符串
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    # ==================== 原有辅助方法 ====================

    async def _handle_delete(self, file_path: str) -> None:
        """
        处理文件删除：从数据库和向量索引中移除

        Args:
            file_path: 文件路径（可以是绝对路径或相对路径）
        """
        db: Session = SessionLocal()
        try:
            # 转换为相对路径（如果需要）
            abs_path = Path(file_path)
            try:
                rel_path = str(abs_path.relative_to(self.config.store_path.parent))
            except ValueError:
                rel_path = str(abs_path)

            # 查找数据库记录
            file_record = db.query(DiaryFileTable).filter(DiaryFileTable.path == rel_path).first()
            if not file_record:
                logging.warning(f"[VectorIndex] File not found in database: {rel_path}")
                return

            # 获取所有 chunk IDs
            chunk_ids = [c.id for c in file_record.chunks]

            # 删除文件记录（cascade 会自动删除 chunks）
            db.delete(file_record)
            db.commit()

            # 从向量索引中删除向量
            if chunk_ids:
                idx = await self._get_or_load_diary_index(file_record.diary_name)
                if idx and hasattr(idx, "remove"):
                    for chunk_id in chunk_ids:
                        try:
                            idx.remove(chunk_id)
                        except Exception as e:
                            logging.warning(f"[VectorIndex] Failed to remove chunk {chunk_id}: {e}")
                    self._schedule_index_save(file_record.diary_name)
                    logging.info(f"[VectorIndex] ✅ Deleted file and removed {len(chunk_ids)} vectors: {rel_path}")
        except Exception as e:
            logging.error(f"[VectorIndex] Delete error: {e}")
            db.rollback()
        finally:
            db.close()

# ==================== 统一同步服务 ====================

async def sync_all_diaries_to_vector_index() -> Dict[str, int]:
    """
    全量同步服务：同步所有角色的日记到向量索引（使用批处理队列）

    适用场景：
    1. 应用首次启动 - 同步所有现有文件到向量索引
    2. 全量修复 - 重建损坏的索引
    3. 手动同步 - 管理员手动触发同步

    注意：文件监视器启动后会自动处理增量变化，无需手动调用

    Returns:
        统计信息字典：{queued, total}
    """
    from app.services.character_service import CharacterService

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("🚀 启动向量索引同步任务")
    logger.info("=" * 60)

    try:
        # 使用已初始化的全局向量索引实例
        vector_index = get_vector_index()

        # 如果尚未初始化，则初始化
        if not hasattr(vector_index, '_initialized') or not vector_index._initialized:
            await vector_index.initialize()
            logger.info("✅ VectorIndex 初始化成功")
        else:
            logger.info("✅ 使用已存在的 VectorIndex 实例")

        # 获取所有角色
        character_service = CharacterService()
        characters = character_service.list_characters()

        if not characters:
            logger.info("📭 没有找到任何角色")
            return {"queued": 0, "total": 0}

        logger.info(f"📚 找到 {len(characters)} 个角色，开始同步日记...")

        # 统计信息
        total_stats = {
            "queued": 0,
            "total": 0
        }

        # 逐个同步角色的日记（添加到批处理队列）
        for character in characters:
            logger.info("-" * 60)
            logger.info(f"📖 处理角色: {character.name} (ID: {character.character_id})")

            result = await vector_index.sync_character_diaries(character.name)

            queued = result.get("queued", 0)
            total = result.get("total", 0)

            logger.info(f"  📥 已加入队列: {queued} 个文件")

            # 累计统计
            total_stats["queued"] += queued
            total_stats["total"] += total

        # 输出总体统计
        logger.info("=" * 60)
        logger.info("📊 向量索引同步完成 - 总体统计")
        logger.info("=" * 60)
        logger.info(f"  已加入队列: {total_stats['queued']} 个文件")
        logger.info(f"  总文件数: {total_stats['total']} 个文件")

        # 手动触发批处理，确保标签被提取和存储
        if vector_index.pending_files:
            logger.info("  🚀 触发批处理...")
            try:
                await vector_index._flush_batch()
                logger.info("  ✅ 批处理完成")
            except Exception as e:
                logger.error(f"  ❌ 批处理失败: {e}", exc_info=True)
        else:
            logger.info("  📭 队列为空，无需批处理")

        logger.info("=" * 60)

        return total_stats

    except Exception as e:
        logger.error(f"❌ 向量索引同步失败: {e}", exc_info=True)
        logger.error("请检查向量索引配置和数据库连接")
        return {"queued": 0, "total": 0}


# ==================== 全局单例 ====================
_vector_index_instance: Optional[VectorIndex] = None


def get_vector_index() -> VectorIndex:
    """
    获取全局 VectorIndex 单例实例

    Returns:
        VectorIndex 实例
    """
    global _vector_index_instance
    if _vector_index_instance is None:
        config = VectorIndexConfig()
        _vector_index_instance = VectorIndex(config)
        logging.info("[VectorIndex] Global instance created")
    return _vector_index_instance


async def initialize_vector_index() -> VectorIndex:
    """
    初始化全局 VectorIndex 实例

    Returns:
        初始化后的 VectorIndex 实例
    """
    vector_index = get_vector_index()
    await vector_index.initialize()
    return vector_index

