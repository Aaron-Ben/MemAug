"""Diary API endpoints for managing character diary entries.

Diaries are stored in data/daily/{name}/

通过 MemoryBackendFactory 统一管理 v1/v2 切换
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from memory import MemoryBackendFactory

logger = logging.getLogger(__name__)

# 获取当前 backend
_current_backend = MemoryBackendFactory.get_backend()
_is_v1 = _current_backend.name == "v1"

# 根据 backend 类型导入对应服务
if _is_v1:
    from memory.v1.services.diary import DiaryFileService
    from memory.v1.plugin_manager import plugin_manager
    from app.models.diary import DiaryEntry
else:
    # V2 模式下 Diary API 不可用
    DiaryFileService = None
    plugin_manager = None
    DiaryEntry = None


def _check_v1_enabled():
    """检查 v1 是否启用，未启用则抛出异常"""
    if _current_backend.name != "v1":
        raise HTTPException(status_code=503, detail="Diary API not available in current memory mode")


def _get_diary_service():
    """获取日记服务实例"""
    if DiaryFileService is None:
        raise HTTPException(status_code=503, detail="Diary API not available in current memory mode")
    return DiaryFileService()


# Create router
router = APIRouter(prefix="/api/v1/diary", tags=["diary"])


# Pydantic models for request/response
class CreateDiaryRequest(BaseModel):
    """创建日记请求"""
    character_id: str = Field(..., description="角色 ID")
    date: str = Field(..., description="日记日期 (YYYY-MM-DD)")
    content: str = Field(..., description="日记内容（应包含末尾的 Tag: xxx）")
    tag: Optional[str] = Field(None, description="可选标签，将添加到内容末尾")


class UpdateDiaryRequest(BaseModel):
    """更新日记请求"""
    content: str = Field(..., description="日记内容（应包含末尾的 Tag 行）")


class AIUpdateDiaryRequest(BaseModel):
    """AI更新日记请求"""
    target: str = Field(..., min_length=15, description="要查找和替换的旧内容（至少15字符）")
    replace: str = Field(..., description="替换的新内容")
    character_id: Optional[str] = Field(None, description="角色 ID（可选，用于指定搜索范围）")


@router.get("/list")
async def list_diaries(character_id: str, limit: int = 10):
    """获取指定角色的日记列表"""
    _check_v1_enabled()
    diary_service = _get_diary_service()
    try:
        diaries = diary_service.list_diaries(character_id=character_id, limit=limit)
        return [DiaryEntry(**d) for d in diaries]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取日记列表失败: {str(e)}")


@router.get("/names")
async def list_diary_names():
    """获取所有有日记的角色 ID 列表"""
    _check_v1_enabled()
    diary_service = _get_diary_service()
    try:
        names = diary_service.list_all_diary_names()
        return {"names": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取角色列表失败: {str(e)}")


@router.get("/latest")
async def get_latest_diary(character_id: str):
    """获取指定角色最新的日记"""
    _check_v1_enabled()
    diary_service = _get_diary_service()
    try:
        diaries = diary_service.list_diaries(character_id=character_id, limit=1)
        if not diaries:
            return None
        return DiaryEntry(**diaries[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取最新日记失败: {str(e)}")


@router.get("/sync")
async def sync_diary_files(character_id: str):
    """同步指定角色的日记文件元数据到数据库"""
    _check_v1_enabled()
    diary_service = _get_diary_service()
    try:
        result = diary_service.update_file_metadata(character_id)
        return {
            "message": "文件元数据同步完成",
            "character_id": character_id,
            "added": result["added"],
            "updated": result["updated"],
            "removed": result["removed"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件同步失败: {str(e)}")


@router.get("/{path:path}")
async def get_diary_by_path(path: str):
    """根据路径获取日记详情"""
    _check_v1_enabled()
    diary_service = _get_diary_service()
    try:
        diary = diary_service.read_diary(path)
        if not diary:
            raise HTTPException(status_code=404, detail="日记不存在")
        return DiaryEntry(**diary)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取日记失败: {str(e)}")


@router.post("/create")
async def create_diary(request: CreateDiaryRequest, background_tasks: BackgroundTasks):
    """AI创建日记（通过 DailyNote 插件）"""
    _check_v1_enabled()
    diary_service = _get_diary_service()

    try:
        # 确保插件已加载
        if not plugin_manager.plugins:
            await plugin_manager.load_plugins()

        # 通过 DailyNote 插件创建日记
        result = await plugin_manager.process_tool_call("DailyNote", {
            "command": "create",
            "maid": request.character_id,
            "date": request.date,
            "content": request.content,
            "tag": request.tag
        })

        if result.get("status") == "success":
            diary_path = result.get("path")
            diary = diary_service.read_diary(diary_path)
            if diary:
                background_tasks.add_task(_trigger_vector_sync)
                return {
                    "message": result.get("message", "日记创建成功"),
                    "diary": DiaryEntry(**diary)
                }
            else:
                return {
                    "message": result.get("message", "日记创建成功"),
                    "diary": DiaryEntry(
                        path=diary_path,
                        character_id=request.character_id,
                        content=request.content,
                        mtime=0
                    )
                }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "创建日记失败"))
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建日记失败: {str(e)}")


@router.post("/ai-update")
async def ai_update_diary(request: AIUpdateDiaryRequest, background_tasks: BackgroundTasks):
    """AI更新日记（通过 DailyNote 插件）"""
    _check_v1_enabled()
    diary_service = _get_diary_service()

    try:
        # 确保插件已加载
        if not plugin_manager.plugins:
            await plugin_manager.load_plugins()

        # 通过 DailyNote 插件更新日记
        result = await plugin_manager.process_tool_call("DailyNote", {
            "command": "update",
            "maid": request.character_id,
            "target": request.target,
            "replace": request.replace
        })

        if result.get("status") == "success":
            diary_path = result.get("path")
            diary = diary_service.read_diary(diary_path)
            if diary:
                if request.character_id:
                    background_tasks.add_task(_trigger_vector_sync)
                return {
                    "message": result.get("message", "日记更新成功"),
                    "path": diary_path,
                    "old_content": request.target,
                    "new_content": request.replace,
                    "diary": DiaryEntry(**diary)
                }
            else:
                return {
                    "message": result.get("message", "日记更新成功"),
                    "path": diary_path,
                    "old_content": request.target,
                    "new_content": request.replace
                }
        else:
            raise HTTPException(status_code=404, detail=result.get("error", "更新日记失败"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新日记失败: {str(e)}")


@router.delete("/{path:path}")
async def delete_diary(path: str):
    """删除日记"""
    _check_v1_enabled()
    diary_service = _get_diary_service()

    try:
        success = diary_service.delete_diary(path)
        if not success:
            raise HTTPException(status_code=404, detail="日记不存在")
        return {"message": "日记删除成功", "path": path}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除日记失败: {str(e)}")


async def _trigger_vector_sync():
    """后台任务：触发向量索引同步"""
    from app.vector_index import sync_all_diaries_to_vector_index

    logger.info("[Diary API] 🚀 触发向量索引同步（后台任务）")
    try:
        result = await sync_all_diaries_to_vector_index()
        logger.info(f"[Diary API] ✅ 向量索引同步完成: {result}")
    except Exception as e:
        logger.error(f"[Diary API] ❌ 向量索引同步失败: {e}", exc_info=True)