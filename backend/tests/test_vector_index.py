"""测试 VectorIndex 批处理方法

这个脚本测试如何为日记文件建立向量索引。
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 添加项目根目录到 Python 路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.vector_index import VectorIndex, VectorIndexConfig
from app.models.database import init_db, SessionLocal, DiaryFileTable, ChunkTable


async def test_sync_all_diaries():
    """测试批量同步日记"""

    print("=" * 60)
    print("批量同步日记测试")
    print("=" * 60)
    print()

    # 初始化数据库
    init_db()

    # 创建 VectorIndex 实例（缩短延迟用于测试）
    config = VectorIndexConfig(batch_delay=2.0)
    vector_index = VectorIndex(config)

    # 同步所有日记
    name = "严肃的老师"

    print(f"同步日记 '{name}' 的所有日记...")
    result = await vector_index.sync_character_diaries(name)

    print()
    print("=" * 60)
    print("同步结果:")
    print("=" * 60)
    print(f"  已加入队列: {result['queued']} 个文件")
    print(f"  总文件数: {result['total']} 个")

    print("\n等待批处理完成...")
    await asyncio.sleep(5)  # 等待批处理完成

    # 检查队列状态
    print(f"  剩余队列: {len(vector_index.pending_files)} 个文件")
    print(f"  处理中: {vector_index.is_processing}")

    print("=" * 60)

    # 保存索引
    await vector_index.flush_all()
    print("✅ 索引已保存")


if __name__ == "__main__":
    import logging

    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 60)
    print("VectorIndex 批处理测试")
    print("=" * 60)
    print()

    # 运行批量同步测试
    asyncio.run(test_sync_all_diaries())
