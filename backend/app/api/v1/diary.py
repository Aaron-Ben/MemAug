"""Diary API endpoints for managing character diary entries.

Diaries are stored in data/characters/{character_id}/daily/
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.diary.file_service import DiaryFileService
from app.models.diary import DiaryEntry
from plugins.plugin import plugin_manager


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


class DiaryResponse(BaseModel):
    """日记响应"""
    diary: DiaryEntry
    message: str


# Dependency injection
def get_diary_file_service():
    """获取日记文件服务实例"""
    return DiaryFileService()


@router.get("/list")
async def list_diaries(
    character_id: str,
    limit: int = 10,
    diary_service: DiaryFileService = Depends(get_diary_file_service)
):
    """
    获取指定角色的日记列表

    Query Parameters:
    - character_id: 角色 ID
    - limit: 返回数量限制 (default: 10)
    """
    try:
        diaries = diary_service.list_diaries(character_id=character_id, limit=limit)
        return [DiaryEntry(**d) for d in diaries]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取日记列表失败: {str(e)}")


@router.get("/names")
async def list_diary_names(
    diary_service: DiaryFileService = Depends(get_diary_file_service)
):
    """获取所有有日记的角色 ID 列表"""
    try:
        names = diary_service.list_all_diary_names()
        return {"names": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取角色列表失败: {str(e)}")


@router.get("/latest")
async def get_latest_diary(
    character_id: str,
    diary_service: DiaryFileService = Depends(get_diary_file_service)
):
    """
    获取指定角色最新的日记

    Query Parameters:
    - character_id: 角色 ID
    """
    try:
        diaries = diary_service.list_diaries(character_id=character_id, limit=1)
        if not diaries:
            return None
        return DiaryEntry(**diaries[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取最新日记失败: {str(e)}")


@router.get("/sync")
async def sync_diary_files(
    character_id: str,
    diary_service: DiaryFileService = Depends(get_diary_file_service)
):
    """同步指定角色的日记文件到数据库"""
    try:
        result = diary_service.sync_character_diaries(character_id)
        return {
            "message": "文件同步完成",
            "character_id": character_id,
            "added": result["added"],
            "updated": result["updated"],
            "removed": result["removed"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件同步失败: {str(e)}")


@router.get("/{path:path}")
async def get_diary_by_path(
    path: str,
    diary_service: DiaryFileService = Depends(get_diary_file_service)
):
    """
    根据路径获取日记详情

    Path Parameters:
    - path: 文件相对路径 (例如: {uuid}/daily/2025-01-23-14_30_52.txt)
    """
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
async def create_diary(
    request: CreateDiaryRequest,
    diary_service: DiaryFileService = Depends(get_diary_file_service)
):
    """
    AI创建日记（通过 DailyNote 插件）

    Request Body:
    ```json
    {
        "character_id": "550e8400-e29b-41d4-a716-446655440000",
        "date": "2025-01-23",
        "content": "今天哥哥陪我玩了一整天...\\n\\nTag: 开心, 约会",
        "tag": null
    }
    ```
    """
    try:
        # 确保插件已加载
        if not plugin_manager.plugins:
            await plugin_manager.load_plugins()

        # 通过 DailyNote 插件创建日记
        result = await plugin_manager.process_tool_call("DailyNote", {
            "command": "create",
            "maid": request.character_id,
            "Date": request.date,
            "Content": request.content,
            "Tag": request.tag
        })

        if result.get("status") == "success":
            # 读取创建的日记并返回
            diary_path = result.get("path")
            diary = diary_service.read_diary(diary_path)
            if diary:
                return {
                    "message": result.get("message", "日记创建成功"),
                    "diary": DiaryEntry(**diary)
                }
            else:
                # 如果读取失败，返回基本信息
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
async def ai_update_diary(
    request: AIUpdateDiaryRequest,
    diary_service: DiaryFileService = Depends(get_diary_file_service)
):
    """
    AI更新日记（通过 DailyNote 插件）

    Request Body:
    ```json
    {
        "target": "这是日记中要被替换掉的旧内容，至少15个字符长",
        "replace": "这是将要写入日记的新内容",
        "character_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    ```

    如果不指定 character_id，会在所有角色的日记中搜索。
    """
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
            # 读取更新后的日记并返回
            diary_path = result.get("path")
            diary = diary_service.read_diary(diary_path)
            if diary:
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
async def delete_diary(
    path: str,
    diary_service: DiaryFileService = Depends(get_diary_file_service)
):
    """
    删除日记

    Path Parameters:
    - path: 文件相对路径

    Returns:
    ```json
    {
        "message": "日记删除成功",
        "path": "550e8400-.../daily/2025-01-23-14_30_52.txt"
    }
    ```
    """
    try:
        success = diary_service.delete_diary(path)
        if not success:
            raise HTTPException(status_code=404, detail="日记不存在")

        return {
            "message": "日记删除成功",
            "path": path
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除日记失败: {str(e)}")
