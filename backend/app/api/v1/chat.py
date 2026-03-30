"""Chat API endpoints for character-based conversations."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime
import logging
import os

from app.services.base_chat_service import BaseChatService
from app.services.llm import LLM
from app.services.character_service import CharacterService
from app.services.chat_history_service import ChatHistoryService
from app.models.character import UserCharacterPreference
from app.schemas.message import ChatRequest, ChatResponse

# Load memory mode
memory_mode = os.getenv("MEMORY", "v1")

# Create router
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Configure logging
logger = logging.getLogger(__name__)

# In-memory storage for user preferences (shared with character API)
from app.api.v1.character import _user_preferences_store


def _create_chat_service(
    llm: LLM,
    character_service: CharacterService,
    history_service: ChatHistoryService,
) -> BaseChatService:
    """工厂函数：根据 memory_mode 创建对应的 ChatService"""
    if memory_mode == "v0":
        from app.services.chat_service_v0 import ChatServiceV0
        return ChatServiceV0(
            llm=llm,
            character_service=character_service,
            history_service=history_service,
        )
    elif memory_mode == "v2":
        from app.services.chat_service_v2 import ChatServiceV2
        return ChatServiceV2(
            llm=llm,
            character_service=character_service,
            history_service=history_service,
        )
    elif memory_mode == "v3":
        from app.services.chat_service_v3 import ChatServiceV3
        from memory.factory import MemoryBackendFactory
        from memory.v3.backend import MemoryV3Backend
        backend = MemoryBackendFactory.get_backend()
        if not isinstance(backend, MemoryV3Backend):
            raise RuntimeError(f"Expected MemoryV3Backend, got {type(backend)}")
        return ChatServiceV3(
            llm=llm,
            character_service=character_service,
            history_service=history_service,
            memory_backend=backend,
        )
    else:
        from app.services.chat_service_v1 import ChatServiceV1
        from memory.v1.plugin_manager import plugin_manager
        return ChatServiceV1(
            llm=llm,
            character_service=character_service,
            history_service=history_service,
            plugin_manager=plugin_manager,
        )


def get_character_service() -> CharacterService:
    """Dependency injection for CharacterService."""
    return CharacterService()


def get_llm_service() -> LLM:
    """Dependency injection for LLM service."""
    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
    return LLM(config={"model": model})


def get_chat_history_service() -> ChatHistoryService:
    """Dependency injection for ChatHistoryService."""
    return ChatHistoryService()


def get_mock_user_id() -> str:
    """Mock user ID for development."""
    return "user_default"


def get_user_preferences(
    character_id: str,
    user_id: str
) -> Optional[UserCharacterPreference]:
    """Get user preferences from store."""
    key = f"{user_id}_{character_id}"
    return _user_preferences_store.get(key)


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_mock_user_id),
    character_service: CharacterService = Depends(get_character_service),
    llm: LLM = Depends(get_llm_service),
    history_service: ChatHistoryService = Depends(get_chat_history_service)
):
    """
    Send a message to a character and get a response.

    Request Body:
    - message: User's message to the character
    - character_id: Character to chat with (UUID)
    - topic_id: Topic ID for continuing a conversation (optional)
    - stream: Whether to stream the response (default: false)
    """
    # Resolve topic_id
    character_id = request.character_id
    topic_id = request.topic_id
    if topic_id is None:
        topic_id = history_service.get_or_create_default_topic(user_id, character_id)

    # Verify character exists
    character = character_service.get_character(character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character not found: {character_id}")
    character_name = character.name if character else character_id

    # Get user preferences
    user_preferences = get_user_preferences(character_id, user_id)

    # Create chat service (version-agnostic)
    chat_service = _create_chat_service(llm, character_service, history_service)

    # Load conversation history
    history_messages = history_service.get_history_for_chat(user_id, topic_id, character_id)
    request_with_history = ChatRequest(
        message=request.message,
        character_id=character_id,
        conversation_history=history_messages if history_messages else None,
        stream=request.stream,
    )

    # Generate response
    try:
        response = await chat_service.chat(request_with_history, user_preferences, user_id)

        # Persist messages
        await chat_service.persist_messages(
            character_id=character_id,
            topic_id=topic_id,
            user_id=user_id,
            character_name=character_name,
            user_message=request.message,
            assistant_message=response.message,
        )

        response.topic_id = topic_id
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    user_id: str = Depends(get_mock_user_id),
    character_service: CharacterService = Depends(get_character_service),
    llm: LLM = Depends(get_llm_service),
    history_service: ChatHistoryService = Depends(get_chat_history_service)
):
    """
    Send a message to a character and get a streaming response (SSE).
    """
    # Resolve topic_id
    character_id = request.character_id
    topic_id = request.topic_id
    if topic_id is None:
        topic_id = history_service.get_or_create_default_topic(user_id, character_id)

    # Verify character exists
    character = character_service.get_character(character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character not found: {character_id}")
    character_name = character.name if character else character_id

    # Get user preferences
    user_preferences = get_user_preferences(character_id, user_id)

    # Create chat service (version-agnostic)
    chat_service = _create_chat_service(llm, character_service, history_service)

    # Load conversation history
    history_messages = history_service.get_history_for_chat(user_id, topic_id, character_id)
    request_with_history = ChatRequest(
        message=request.message,
        character_id=character_id,
        conversation_history=history_messages if history_messages else None,
        stream=request.stream,
    )

    # Store full response for persistence
    full_response = []

    async def generate():
        try:
            async for chunk in chat_service.chat_stream(request_with_history, user_preferences, user_id):
                full_response.append(chunk)
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

            # Persist messages after stream completes
            response_text = "".join(full_response)
            await chat_service.persist_messages(
                character_id=character_id,
                topic_id=topic_id,
                user_id=user_id,
                character_name=character_name,
                user_message=request.message,
                assistant_message=response_text,
            )
        except Exception as e:
            yield f"data: [ERROR: {str(e)}]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ─── Session lifecycle (V3 only) ─────────────────────────────

class SessionCloseRequest(BaseModel):
    character_id: str
    topic_id: int


@router.post("/session/close")
async def close_session(
    request: SessionCloseRequest,
    user_id: str = Depends(get_mock_user_id),
):
    """结束会话，触发 V3 图谱维护（仅 v3 模式可用）"""
    if memory_mode != "v3":
        raise HTTPException(status_code=400, detail="Session close only supported in v3 mode")

    from memory.factory import MemoryBackendFactory
    from memory.v3.backend import MemoryV3Backend
    backend = MemoryBackendFactory.get_backend()
    if not isinstance(backend, MemoryV3Backend):
        raise HTTPException(status_code=500, detail="Memory backend is not V3")

    try:
        result = await backend.finalize_session(request.character_id, str(request.topic_id))
        return {"status": "finalized", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Finalize failed: {str(e)}")


# ─── Graph stats (V3 only) ────────────────────────────────────

class GraphStatsRequest(BaseModel):
    character_id: str


@router.post("/graph/stats")
async def graph_stats(
    request: GraphStatsRequest,
    user_id: str = Depends(get_mock_user_id),
):
    """获取 V3 图谱统计（仅 v3 模式可用）"""
    if memory_mode != "v3":
        raise HTTPException(status_code=400, detail="Graph stats only supported in v3 mode")

    from memory.factory import MemoryBackendFactory
    from memory.v3.backend import MemoryV3Backend
    backend = MemoryBackendFactory.get_backend()
    if not isinstance(backend, MemoryV3Backend):
        raise HTTPException(status_code=500, detail="Memory backend is not V3")

    try:
        stats = await backend.get_graph_stats(request.character_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


# ─── Logs ──────────────────────────────────────────────────────

@router.get("/logs/today")
async def get_today_logs():
    """Get today's chat and tool call logs."""
    try:
        from app.utils.file_logger import get_log_content
        from datetime import datetime

        log_content = get_log_content()
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "content": log_content,
            "lines": len(log_content.split('\n')) if log_content else 0
        }
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")


@router.get("/logs/list")
async def list_logs():
    """List all available log files."""
    try:
        from app.utils.file_logger import list_log_files

        log_files = list_log_files()
        return {
            "logs": [
                {"filename": filename, "date": date_str}
                for filename, date_str in log_files
            ]
        }
    except Exception as e:
        logger.error(f"Error listing logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing logs: {str(e)}")


@router.get("/logs/{date}")
async def get_logs_by_date(date: str):
    """Get logs for a specific date (YYYY-MM-DD)."""
    try:
        from app.utils.file_logger import get_log_content

        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        log_content = get_log_content(datetime.strptime(date, "%Y-%m-%d"))

        if not log_content:
            raise HTTPException(status_code=404, detail=f"No logs found for date: {date}")

        return {
            "date": date,
            "content": log_content,
            "lines": len(log_content.split('\n'))
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading logs for {date}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")
