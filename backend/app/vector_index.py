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

# ==================== Database ====================
from sqlalchemy.orm import Session
from app.models.database import DiaryFileTable, ChunkTable, TagTable, FileTagTable, KVStoreTable, SessionLocal

# ==================== Services ====================
from app.services.chunk_text import chunk_text
from app.services.embedding import EmbeddingService
from app.services.character_service import CharacterService

# ==================== 配置 ====================
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

    @property
    def store_path(self) -> Path:
        """获取VectorStore目录路径（项目根目录）"""
        # 从 backend/app/ 向上两级到达项目根目录
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        return project_root / "VectorStore"

    @property
    def data_daily_path(self) -> Path:
        """获取 data/daily 目录路径"""
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        return project_root / "data" / "daily"


# ==================== VexusIndex 导入 ====================
from vector_db import VexusIndex


# ==================== 搜索结果数据类 ====================
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
    core_tags_matched: List[str] = field(default_factory=list) # 核心标签匹配


# ==================== Tag 增强结果 ====================
@dataclass
class TagBoostResult:
    """Tag 增强结果"""
    vector: List[float]                # 增强后的向量
    info: Optional[Dict[str, Any]]     # 增强信息（匹配的标签等）


# ==================== 核心管理类 ====================
class VectorIndex:
    """向量索引管理器 - 支持多索引、Tag 搜索和增强"""

    def __init__(self, config: VectorIndexConfig):
        self.config = config
        self.diary_indices: Dict[str, VexusIndex] = {}  # diaryName -> VexusIndex实例
        self.tag_index: Optional[VexusIndex] = None     # 全局 Tag 索引
        self.save_tasks: Dict[str, Optional[asyncio.Task]] = {}  # diaryName -> asyncio.Task
        self.diary_name_vector_cache: Dict[str, List[float]] = {}  # 日记本名称向量缓存

        # 批处理队列
        self.pending_files: Set[str] = set()  # 待处理文件集合
        self.file_retry_count: Dict[str, int] = {}  # 文件重试计数
        self.is_processing: bool = False  # 是否正在处理
        self.batch_timer: Optional[asyncio.Task] = None  # 批处理定时器

        # 文件监视器
        self.watcher = None  # watchdog.Observer 实例
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None  # 主事件循环

        self._ensure_store_path()

    def _ensure_store_path(self) -> None:
        """确保存储目录存在"""
        store_path = self.config.store_path
        if not store_path.exists():
            store_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"[VectorIndex] Created store path: {store_path}")

    # ==================== 初始化 ====================
    async def initialize(self) -> None:
        """
        初始化向量索引系统
        - 初始化全局 Tag 索引
        - 预热日记本名称向量缓存
        - 启动文件监视器
        """
        logging.info("[VectorIndex] 🚀 Initializing Multi-Index System...")

        # 保存事件循环引用（用于监视器回调）
        self.event_loop = asyncio.get_running_loop()

        # 1. 初始化全局 Tag 索引
        await self._init_tag_index()

        # 2. 预热日记本名称向量缓存
        self._hydrate_diary_name_cache()

        # 3. 启动文件监视器
        if self.config.enable_watcher:
            self._start_watcher()

        logging.info("[VectorIndex] ✅ System Ready")

    async def _init_tag_index(self) -> None:
        """初始化全局 Tag 索引"""
        tag_idx_path = self.config.store_path / "index_global_tags.usearch"

        try:
            if tag_idx_path.exists():
                logging.info("[VectorIndex] 📂 Loading existing Tag index...")
                self.tag_index = VexusIndex.load(
                    self.config.dimension,
                    self.config.capacity,
                    str(tag_idx_path)
                )
                logging.info("[VectorIndex] ✅ Tag index loaded from disk")
            else:
                logging.info("[VectorIndex] ✨ Creating new Tag index...")
                self.tag_index = VexusIndex(self.config.dimension, self.config.capacity)
                # 从数据库恢复标签
                await self._recover_tags_from_db()
                await self._save_index_to_disk()
        except Exception as e:
            logging.error(f"[VectorIndex] ❌ Tag index load failed: {e}")
            logging.warning("[VectorIndex] 🔄 Creating new Tag index as fallback...")
            self.tag_index = VexusIndex(self.config.dimension, self.config.capacity)
            await self._recover_tags_from_db()

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
                logging.info(f"[VectorIndex] Index file not found for {file_name}, creating a new empty one.")
                idx = VexusIndex(self.config.dimension, self.config.capacity)
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
        从数据库恢复向量数据到索引（使用 Rust 高性能恢复）

        Args:
            idx: 索引实例
            diary_name: 日记本名称
        """
        logging.info(f"[VectorIndex] 🔄 Recovering chunks for \"{diary_name}\" via Rust...")
        try:
            db_path = self.config.store_path / "emotional_companionship.db"
            count = idx.recover_from_sqlite(str(db_path), table_type, diary_name)
            logging.info(f"[VectorIndex] ✅ Recovered {count} vectors via Rust")
        except Exception as e:
            logging.error(f"[VectorIndex] ❌ Rust recovery failed: {e}")

    async def _recover_tags_from_db(self) -> None:
        """从数据库恢复标签到索引（使用 Rust 高性能恢复）"""
        logging.info("[VectorIndex] 🔄 Recovering Tag index from database via Rust...")
        try:
            db_path = self.config.store_path / "emotional_companionship.db"
            count = self.tag_index.recover_from_sqlite(str(db_path), "tags", None)
            logging.info(f"[VectorIndex] ✅ Recovered {count} tags via Rust")
            await self._save_index_to_disk("global_tags")
        except Exception as e:
            logging.error(f"[VectorIndex] ❌ Rust recovery failed: {e}")


    # ==================== 核心搜索接口 ====================
    async def search(
        self,
        arg1: Union[str, List[float]],
        arg2: Optional[Union[List[float], int]] = None,
        arg3: Optional[int] = 5,
        arg4: Optional[float] = 0,
        arg5: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        统一搜索接口 - 支持多种参数组合

        参数组合：
        1. search(diary_name, vector, k, tag_boost, core_tags)
        2. search(vector, k, tag_boost, core_tags)
        3. search(query_text, k) - 需要先向量化

        Args:
            arg1: 日记本名称 或 向量 或 查询文本
            arg2: 向量 或 k 值
            arg3: k 值 或 tag_boost
            arg4: tag_boost 或 core_tags
            arg5: core_tags

        Returns:
            搜索结果列表
        """
        try:
            diary_name = None
            query_vec = None
            k = 5
            tag_boost = 0
            core_tags = []

            # 解析参数
            if isinstance(arg1, str) and isinstance(arg2, list):
                # search(diary_name, vector, k, tag_boost, core_tags)
                diary_name = arg1
                query_vec = arg2
                k = arg3 if isinstance(arg3, int) else 5
                tag_boost = arg4 if isinstance(arg4, (int, float)) else 0
                core_tags = arg5 if arg5 else []
            elif isinstance(arg1, str):
                # 纯字符串查询，返回空（需要先向量化）
                return []
            elif isinstance(arg1, list):
                # search(vector, k, tag_boost, core_tags)
                query_vec = arg1
                k = arg2 if isinstance(arg2, int) else 5
                tag_boost = arg3 if isinstance(arg3, (int, float)) else 0
                core_tags = arg4 if isinstance(arg4, list) else []

            if not query_vec:
                return []

            if diary_name:
                return await self._search_specific_index(diary_name, query_vec, k, tag_boost, core_tags or [])
            else:
                return await self._search_all_indices(query_vec, k, tag_boost, core_tags or [])

        except Exception as e:
            logging.error(f"[VectorIndex] Search error: {e}")
            return []

    async def _search_specific_index(
        self,
        diary_name: str,
        vector: List[float],
        k: int,
        tag_boost: float,
        core_tags: List[str]
    ) -> List[SearchResult]:
        """
        在指定日记本索引中搜索

        Args:
            diary_name: 日记本名称
            vector: 查询向量
            k: 返回结果数量
            tag_boost: 标签增强因子
            core_tags: 核心标签列表

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
        if tag_boost > 0 and core_tags:
            boost_result = self._apply_simple_tag_boost(vector, tag_boost, core_tags)
            search_vec = boost_result.vector
            tag_info = boost_result.info

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
                        matched_tags=tag_info.matched_tags if tag_info else [],
                        boost_factor=tag_info.boost_factor if tag_info else 0,
                        core_tags_matched=tag_info.core_tags_matched if tag_info else []
                    ))
            return search_results
        finally:
            db.close()

    async def _search_all_indices(
        self,
        vector: List[float],
        k: int,
        tag_boost: float,
        core_tags: List[str]
    ) -> List[SearchResult]:
        """
        在所有日记本索引中搜索

        Args:
            vector: 查询向量
            k: 返回结果数量
            tag_boost: 标签增强因子
            core_tags: 核心标签列表

        Returns:
            搜索结果列表
        """
        # 应用 Tag 增强（如果启用）
        search_vec = vector
        tag_info = None
        if tag_boost > 0 and core_tags:
            boost_result = self._apply_simple_tag_boost(vector, tag_boost, core_tags)
            search_vec = boost_result.vector
            tag_info = boost_result.info

        search_buffer = self._serialize_vector(search_vec)

        # 获取所有日记本名称
        db: Session = SessionLocal()
        try:
            all_diaries = db.query(DiaryFileTable.diary_name).distinct().all()
            diary_names = [d.diary_name for d in all_diaries]
        finally:
            db.close()

        # 并行搜索所有索引
        search_tasks = []
        for diary_name in diary_names:
            search_tasks.append(self._search_single_index(diary_name, search_buffer, k))

        results_per_index = await asyncio.gather(*search_tasks, return_exceptions=True)

        # 合并并排序结果
        all_results = []
        for results in results_per_index:
            if isinstance(results, list):
                all_results.extend(results)

        # 按分数排序，取前 k 个
        all_results.sort(key=lambda r: r.score, reverse=True)
        top_k = all_results[:k]

        # 更新 Tag 信息
        if tag_info:
            for result in top_k:
                result.matched_tags = tag_info.matched_tags
                result.boost_factor = tag_info.boost_factor
                result.core_tags_matched = tag_info.core_tags_matched

        return top_k

    async def _search_single_index(
        self,
        diary_name: str,
        search_buffer: bytes,
        k: int
    ) -> List[Tuple[int, float]]:
        """搜索单个索引，返回 (chunk_id, score) 列表"""
        try:
            idx = await self._get_or_load_diary_index(diary_name)
            stats = idx.stats()
            if stats.total_vectors == 0:
                return []
            return idx.search(search_buffer, k)
        except Exception as e:
            logging.error(f"[VectorIndex] Search error in \"{diary_name}\": {e}")
            return []

    def _apply_simple_tag_boost(
        self,
        vector: List[float],
        base_tag_boost: float,
        core_tags: List[str]
    ) -> TagBoostResult:
        """
        简化的 Tag 增强：仅基于核心标签的加权融合

        Args:
            vector: 原始查询向量
            base_tag_boost: 增强因子 (0-1)
            core_tags: 核心标签列表

        Returns:
            TagBoostResult 包含增强后的向量和标签信息
        """
        dim = len(vector)

        try:
            # 过滤有效的核心标签
            safe_core_tags = [t for t in core_tags if isinstance(t, str)]
            if not safe_core_tags:
                return TagBoostResult(vector=vector, info=None)

            # 从数据库查询标签向量
            db: Session = SessionLocal()
            try:
                tag_rows = db.query(TagTable).filter(TagTable.name.in_(safe_core_tags)).all()
            finally:
                db.close()

            if not tag_rows:
                return TagBoostResult(vector=vector, info=None)

            # 构建上下文向量
            context_vec = [0.0] * dim
            matched_tags = []

            for row in tag_rows:
                if row.vector:
                    try:
                        v = json.loads(row.vector)
                        for d in range(dim):
                            context_vec[d] += v[d]
                        matched_tags.append(row.name)
                    except (json.JSONDecodeError, TypeError):
                        pass

            if not matched_tags:
                return TagBoostResult(vector=vector, info=None)

            # 归一化上下文向量
            magnitude = sum(x * x for x in context_vec) ** 0.5
            if magnitude > 1e-9:
                context_vec = [x / magnitude for x in context_vec]
            else:
                # 计算平均值
                for d in range(dim):
                    context_vec[d] /= len(matched_tags)
                magnitude = sum(x * x for x in context_vec) ** 0.5
                if magnitude > 1e-9:
                    context_vec = [x / magnitude for x in context_vec]

            # 融合原始向量和上下文向量
            fused = [
                (1 - base_tag_boost) * vector[d] + base_tag_boost * context_vec[d]
                for d in range(dim)
            ]

            # 归一化结果
            magnitude = sum(x * x for x in fused) ** 0.5
            if magnitude > 1e-9:
                fused = [x / magnitude for x in fused]

            return TagBoostResult(
                vector=fused,
                info={
                    "matchedTags": matched_tags,
                    "coreTagsMatched": matched_tags,
                    "boostFactor": base_tag_boost
                }
            )

        except Exception as e:
            logging.error(f"[VectorIndex] Tag boost failed: {e}")
            return TagBoostResult(vector=vector, info=None)

    def apply_tag_boost(
        self,
        vector: List[float],
        tag_boost: float,
        core_tags: List[str]
    ) -> TagBoostResult:
        """
        公共接口：应用 Tag 增强

        Args:
            vector: 原始向量
            tag_boost: 增强因子
            core_tags: 核心标签列表

        Returns:
            TagBoostResult
        """
        return self._apply_simple_tag_boost(vector, tag_boost, core_tags)

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
        logging.info(f"[VectorIndex] ➕➕ Adding {len(vectors)} vectors to diary \"{diary_name}\"")

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
        logging.info(
            f"[VectorIndex] ⏰ Scheduled save for \"{diary_name}\" "
            f"in {self.config.index_save_delay}s"
        )

    async def _save_index_to_disk(self, diary_name: str) -> None:
        """
        立即保存索引到磁盘

        Args:
            diary_name: 日记本名称或 'global_tags'
        """
        try:
            if diary_name == 'global_tags':
                # 保存全局标签索引
                file_path = self.config.store_path / 'index_global_tags.usearch'
                self.tag_index.save(str(file_path))
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

    async def get_diary_name_vector(self, diary_name: str) -> Optional[List[float]]:
        """
        获取日记本名称的向量（带缓存）

        Args:
            diary_name: 日记本名称

        Returns:
            向量列表或 None
        """
        if not diary_name:
            return None

        # 检查缓存
        if diary_name in self.diary_name_vector_cache:
            return self.diary_name_vector_cache[diary_name]

        # 从数据库查找
        db: Session = SessionLocal()
        try:
            entry = db.query(KVStoreTable).filter(
                KVStoreTable.key == f"diary_name:{diary_name}"
            ).first()
            if entry and entry.vector:
                vector = json.loads(entry.vector)
                self.diary_name_vector_cache[diary_name] = vector
                return vector
        finally:
            db.close()

        # 缓存未命中，获取新向量
        logging.warning(f"[VectorIndex] Cache MISS for diary name: \"{diary_name}\". Fetching now...")
        return await self._fetch_and_cache_diary_name_vector(diary_name)

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

    async def get_plugin_description_vector(
        self,
        desc_text: str,
        get_embedding_fn: Optional[callable] = None
    ) -> Optional[List[float]]:
        """
        获取插件描述的向量（带缓存）

        Args:
            desc_text: 描述文本
            get_embedding_fn: 可选的自定义向量化函数

        Returns:
            向量列表或 None
        """
        try:
            # 使用哈希作为缓存键
            import hashlib as hl
            hash_key = hl.sha256(desc_text.encode()).hexdigest()
            cache_key = f"plugin_desc_hash:{hash_key}"

            # 检查数据库缓存
            db: Session = SessionLocal()
            try:
                entry = db.query(KVStoreTable).filter(KVStoreTable.key == cache_key).first()
                if entry and entry.vector:
                    return json.loads(entry.vector)
            finally:
                db.close()

            # 缓存未命中，使用提供的函数或默认服务
            if get_embedding_fn and callable(get_embedding_fn):
                vector = await get_embedding_fn(desc_text)
            else:
                async with EmbeddingService() as embedding_service:
                    vectors = await embedding_service.get_embeddings_batch([desc_text])
                    vector = vectors[0] if vectors else None

            if vector:
                self._save_kv_store(cache_key, vector)
                return vector

        except Exception as e:
            logging.error(f"[VectorIndex] Failed to get plugin description vector: {e}")

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
        # 移除装饰性 emoji
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002600-\U000026FF"  # misc symbols
            "\U00002700-\U000027BF"  # dingbats
            "]+",
            flags=re.UNICODE
        )
        cleaned = emoji_pattern.sub(' ', text)
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
        if auto_schedule and self.event_loop and asyncio.get_event_loop() == self.event_loop:
            self._schedule_batch_flush()
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

                        if existing and existing.checksum == checksum:
                            # 文件未变化，跳过
                            self.pending_files.discard(key)
                            self.file_retry_count.pop(key, None)
                            skipped_files.append({
                                "character_name": character_name,
                                "file_path": file_path
                            })
                            continue

                        # 获取角色信息
                        character_service = CharacterService()
                        character = character_service.get_character_by_name(character_name)
                        if not character:
                            logging.warning(f"[VectorIndex] ❌ Character not found: {character_name}")
                            self.pending_files.discard(key)
                            self.file_retry_count.pop(key, None)
                            continue

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
            tag_updates = []

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

            # 清理已处理的文件
            for key in batch_keys:
                if key in self.pending_files:
                    self.pending_files.discard(key)
                self.file_retry_count.pop(key, None)

            logging.info(f"[VectorIndex] ✅ Batch complete. Updated {len(updates_by_diary)} diary indices.")

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

        return {"queued": len(txt_files), "total": len(txt_files)}

    async def rebuild_index_from_db(
        self,
        diary_name: str
    ) -> None:
        """
        从数据库重建向量索引

        Args:
            diary_name: 日记本名称（角色名称）
        """
        logging.info(f"[VectorIndex] Rebuilding index for diary: {diary_name}")

        db: Session = SessionLocal()
        try:
            # 查询所有chunks
            chunks = db.query(ChunkTable).join(DiaryFileTable).filter(
                DiaryFileTable.diary_name == diary_name
            ).order_by(ChunkTable.chunk_index).all()

            if not chunks:
                logging.warning(f"[VectorIndex] No chunks found for diary: {diary_name}")
                return

            # 准备向量数据
            vector_tuples = []
            for chunk in chunks:
                try:
                    vector = json.loads(chunk.vector)
                    vector_bytes = self._serialize_vector(vector)
                    # 使用 chunk.id 作为向量ID
                    vector_tuples.append((chunk.id, vector_bytes))
                except (json.JSONDecodeError, TypeError) as e:
                    logging.warning(f"[VectorIndex] Failed to parse vector for chunk {chunk.id}: {e}")

            if vector_tuples:
                # 批量添加向量（会自动触发索引加载）
                await self.add_vectors(diary_name, vector_tuples)
                logging.info(f"[VectorIndex] Rebuilt index with {len(vector_tuples)} vectors")

        finally:
            db.close()

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

    async def _vectorize_chunks(
        self,
        chunks: List[str]
    ) -> List[Optional[List[float]]]:
        """
        批量向量化文本块

        Args:
            chunks: 文本块列表

        Returns:
            向量列表（失败的项为None）
        """
        if not chunks:
            return []

        try:
            async with EmbeddingService() as embedding_service:
                vectors = await embedding_service.get_embeddings_batch(chunks)
                return vectors
        except Exception as e:
            logging.error(f"[VectorIndex] Vectorization failed: {e}")
            return [None] * len(chunks)

    async def _process_file_tags(
        self,
        db: Session,
        file_id: int,
        tags: List[str]
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

                        # 添加到全局 Tag 索引
                        if self.tag_index:
                            vector_bytes = self._serialize_vector(vector)
                            try:
                                self.tag_index.add(new_tag.id, vector_bytes)
                            except Exception as e:
                                if "Duplicate" in str(e):
                                    if hasattr(self.tag_index, "remove"):
                                        self.tag_index.remove(new_tag.id)
                                    self.tag_index.add(new_tag.id, vector_bytes)
            except Exception as e:
                logging.error(f"[VectorIndex] Failed to vectorize tags: {e}")

        # 创建文件-标签关联
        for tag, tag_id in tag_cache.items():
            if tag_id is not None:
                association = FileTagTable(file_id=file_id, tag_id=tag_id)
                db.add(association)

        # 安排保存 Tag 索引
        self._schedule_tag_index_save()

    def _schedule_tag_index_save(self) -> None:
        """安排 Tag 索引延迟保存"""
        if "global_tags" in self.save_tasks and self.save_tasks["global_tags"] is not None:
            self.save_tasks["global_tags"].cancel()

        async def save_task():
            await asyncio.sleep(self.config.tag_index_save_delay)
            await self._save_index_to_disk()
            self.save_tasks["global_tags"] = None

        task = asyncio.create_task(save_task())
        self.save_tasks["global_tags"] = task

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
        project_root = current_file.parent.parent.parent
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

    def stats(self) -> Dict[str, Dict[str, int]]:
        """
        获取所有已加载索引的统计信息

        Returns:
            字典，key为日记本名称，value为统计信息字典
        """
        result = {}
        for diary_name, idx in self.diary_indices.items():
            stats_obj = idx.stats()
            result[diary_name] = {
                "totalVectors": stats_obj.total_vectors,
                "dimensions": stats_obj.dimensions,
                "capacity": stats_obj.capacity,
                "memoryUsage": stats_obj.memory_usage,
            }
        return result

    def get_stats(self, diary_name: str) -> Optional[Dict[str, int]]:
        """
        获取索引统计信息

        Args:
            diary_name: 日记本名称

        Returns:
            统计信息字典，如果索引不存在则返回None
        """
        idx = self.diary_indices.get(diary_name)
        if idx is None:
            return None
        stats_obj = idx.stats()
        return {
            "totalVectors": stats_obj.total_vectors,
            "dimensions": stats_obj.dimensions,
            "capacity": stats_obj.capacity,
            "memoryUsage": stats_obj.memory_usage,
        }

    async def flush_all(self) -> None:
        """立即保存所有待保存的索引"""
        logging.info("[VectorIndex] 💾💾 Flushing all pending saves...")
        for diary_name, task in list(self.save_tasks.items()):
            if task is not None:
                task.cancel()
                del self.save_tasks[diary_name]
                await self._save_index_to_disk(diary_name)
        logging.info("[VectorIndex] ✅ All indices saved.")

    def get_index_path(self, diary_name: str) -> Path:
        """
        获取索引文件路径（供调试用）

        Args:
            diary_name: 日记本名称

        Returns:
            索引文件路径
        """
        safe_name = hashlib.md5(diary_name.encode()).hexdigest()
        return self.config.store_path / f"index_diary_{safe_name}.usearch"

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
        # 初始化向量索引
        config = VectorIndexConfig()
        vector_index = VectorIndex(config)
        logger.info("✅ VectorIndex 初始化成功")

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
        logger.info("  ⏳ 批处理将在后台自动执行...")
        logger.info("=" * 60)

        return total_stats

    except Exception as e:
        logger.error(f"❌ 向量索引同步失败: {e}", exc_info=True)
        logger.error("请检查向量索引配置和数据库连接")
        return {"queued": 0, "total": 0}
    
