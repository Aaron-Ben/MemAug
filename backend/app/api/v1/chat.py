"""Chat API endpoints for character-based conversations."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from datetime import datetime
import logging
import os

from app.services.llm import LLM
from app.services.character_service import CharacterService
from app.services.chat_history_service import ChatHistoryService
from app.models.character import UserCharacterPreference
from app.schemas.message import ChatRequest, ChatResponse

# Load ChatService based on MEMORY env
memory_mode = os.getenv("MEMORY", "v1")
if memory_mode == "v0":
    from app.services.chat_service_v0 import ChatServiceV0 as ChatService
elif memory_mode == "v2":
    from app.services.chat_service_v2 import ChatServiceV2 as ChatService
    from app.services.session_service import SessionService
    from memory.v2.chromadb_manager import ChromaDBManager
    from memory.v2.retriever import HierarchicalRetriever
    # v2 模式下不使用 plugin_manager
    plugin_manager = None
else:
    from app.services.chat_service_v1 import ChatService
    # v1 模式：导入 plugin_manager
    from memory.v1.plugin_manager import plugin_manager

# Create router
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Configure logging
logger = logging.getLogger(__name__)

# In-memory storage for user preferences (shared with character API)
from app.api.v1.character import _user_preferences_store


def get_character_service() -> CharacterService:
    """Dependency injection for CharacterService."""
    return CharacterService()


def get_llm_service() -> LLM:
    """
    Dependency injection for LLM service.
    Uses OpenRouter by default.

    Note: Requires OPENROUTER_API_KEY environment variable.
    """
    # Get model from environment or use default
    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

    return LLM(config={"model": model})


def get_chat_history_service() -> ChatHistoryService:
    """Dependency injection for ChatHistoryService."""
    return ChatHistoryService()


def get_mock_user_id() -> str:
    """
    Mock user ID for development.
    In production, this would come from authentication.
    """
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

    Tool calling is handled automatically by chat_service.chat.
    The response will not contain tool call markers <<<[TOOL_REQUEST]>>>.

    Request Body:
    - message: User's message to the character
    - character_id: Character to chat with (UUID)
    - topic_id: Topic ID for continuing a conversation (optional)
    - conversation_history: Optional previous messages for context (deprecated, use topic_id)
    - stream: Whether to stream the response (default: false)

    Returns:
        Character's response with metadata including topic_id

    Example:
    ```json
    {
        "message": "我回来了",
        "character_id": "550e8400-e29b-41d4-a716-446655440000",
        "stream": false
    }
    ```
    """
    # Resolve topic_id (get or create default if not provided)
    character_id = request.character_id
    topic_id = request.topic_id
    if topic_id is None:
        topic_id = history_service.get_or_create_default_topic(user_id, character_id)

    # Verify character exists and get character name
    character = character_service.get_character(character_id)
    if not character:
        raise HTTPException(
            status_code=404,
            detail=f"Character not found: {character_id}"
        )
    character_name = character.name if character else character_id

    # Get user preferences if available
    user_preferences = get_user_preferences(character_id, user_id)

    # Initialize services for v2 mode
    session_service = None
    retriever = None
    if memory_mode == "v2":
        chromadb_manager = ChromaDBManager()
        session_service = SessionService(chromadb_manager=chromadb_manager)
        # 初始化记忆检索器
        from app.services.embedding import EmbeddingService
        embedding_service = EmbeddingService()
        retriever = HierarchicalRetriever(
            chromadb_manager=chromadb_manager,
            embedding_service=embedding_service,
        )

    # Load conversation history from topic
    history_messages = history_service.get_history_for_chat(user_id, topic_id, character_id)

    # Create chat service based on memory mode
    if memory_mode == "v0":
        chat_service = ChatService(
            llm=llm,
            character_service=character_service
        )
    elif memory_mode == "v2":
        chat_service = ChatService(
            llm=llm,
            character_service=character_service
        )
    else:
        chat_service = ChatService(
            llm=llm,
            character_service=character_service,
            plugin_manager=plugin_manager
        )

    # Generate response
    try:
        # Ensure plugins are loaded
        # Load plugins only for v1 mode (v2 uses SessionService)
        if memory_mode == "v1":
            if not plugin_manager.plugins:
                await plugin_manager.load_plugins()

        # Create modified request with history
        request_with_history = ChatRequest(
            message=request.message,
            character_id=character_id,
            conversation_history=history_messages if history_messages else None,
            stream=request.stream
        )

        # Retrieve relevant memories for v2 mode
        memory_context = ""
        if memory_mode == "v2" and retriever:
            try:
                from memory.v2.retriever import SpaceType
                result = await retriever.retrieve(
                    query=request.message,
                    user=user_id,
                    space=SpaceType.USER,
                    limit=5
                )
                if result.matched_contexts:
                    memory_parts = ["[相关记忆参考]"]
                    for ctx in result.matched_contexts:
                        content = ctx.abstract[:300] if len(ctx.abstract) > 300 else ctx.abstract
                        memory_parts.append(f"- {content}")
                    memory_context = "\n".join(memory_parts)
                    logger.info(f"[Memory] Retrieved {len(result.matched_contexts)} contexts")
            except Exception as e:
                logger.warning(f"[Memory] Failed to retrieve: {e}")

        # Use chat method (tool calling is handled internally via chat_stream)
        if memory_mode == "v2":
            response = await chat_service.chat(
                request=request_with_history,
                user_preferences=user_preferences,
                user_id=user_id,
                memory_context=memory_context,
                session_service=session_service,
                retriever=retriever
            )
        else:
            response = await chat_service.chat(
                request=request_with_history,
                user_preferences=user_preferences,
                user_id=user_id
            )

        # Save messages based on memory mode
        if memory_mode == "v2" and session_service:
            # Use SessionService for v2 mode (handles auto-commit)
            await session_service.add_message(
                character_id=character_id,
                topic_id=topic_id,
                role="user",
                content=request.message,
                name=user_id,
                user_id=user_id
            )
            await session_service.add_message(
                character_id=character_id,
                topic_id=topic_id,
                role="assistant",
                content=response.message,
                name=character_name,
                user_id=user_id
            )
        else:
            # Use history_service for v0/v1 mode
            history_service.append_message(
                user_id=user_id,
                topic_id=topic_id,
                role="user",
                content=request.message,
                name=user_id,
                character_id=character_id
            )
            history_service.append_message(
                user_id=user_id,
                topic_id=topic_id,
                role="assistant",
                content=response.message,
                name=character_name,
                character_id=character_id
            )

        # Update response with topic information
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
    Send a message to a character and get a streaming response.

    Tool calling is handled automatically by chat_service.chat_stream.
    The response will not contain tool call markers <<<[TOOL_REQUEST]>>>.

    Returns:
        Server-Sent Events (SSE) stream with response chunks

    Example:
    ```json
    {
        "message": "我回来了",
        "character_id": "550e8400-e29b-41d4-a716-446655440000",
        "stream": true
    }
    ```
    """
    # Resolve topic_id (get or create default if not provided)
    character_id = request.character_id
    topic_id = request.topic_id
    if topic_id is None:
        topic_id = history_service.get_or_create_default_topic(user_id, character_id)

    # Verify character exists and get character name
    character = character_service.get_character(character_id)
    if not character:
        raise HTTPException(
            status_code=404,
            detail=f"Character not found: {character_id}"
        )
    character_name = character.name if character else character_id

    # Get user preferences if available
    user_preferences = get_user_preferences(character_id, user_id)

    # Initialize services for v2 mode
    session_service = None
    retriever = None
    if memory_mode == "v2":
        chromadb_manager = ChromaDBManager()
        session_service = SessionService(chromadb_manager=chromadb_manager)
        # 初始化记忆检索器
        from app.services.embedding import EmbeddingService
        embedding_service = EmbeddingService()
        retriever = HierarchicalRetriever(
            chromadb_manager=chromadb_manager,
            embedding_service=embedding_service,
        )

    # Load conversation history from topic
    history_messages = history_service.get_history_for_chat(user_id, topic_id, character_id)

    # Create chat service based on memory mode
    if memory_mode == "v0":
        chat_service = ChatService(
            llm=llm,
            character_service=character_service
        )
    elif memory_mode == "v2":
        chat_service = ChatService(
            llm=llm,
            character_service=character_service
        )
    else:
        chat_service = ChatService(
            llm=llm,
            character_service=character_service,
            plugin_manager=plugin_manager
        )

    # Store full response for diary generation and saving
    full_response = []

    async def generate():
        """Generate streaming response with tool calling support."""
        try:
            # Ensure plugins are loaded
            # Load plugins only for v1 mode (v2 uses SessionService)
            if memory_mode == "v1":
                if not plugin_manager.plugins:
                    await plugin_manager.load_plugins()

            # Create request with history for building messages
            request_with_history = ChatRequest(
                message=request.message,
                character_id=character_id,
                conversation_history=history_messages if history_messages else None,
                stream=request.stream
            )

            # Retrieve relevant memories for v2 mode
            memory_context = ""
            if memory_mode == "v2" and retriever:
                try:
                    from memory.v2.retriever import SpaceType
                    result = await retriever.retrieve(
                        query=request.message,
                        user=user_id,
                        space=SpaceType.USER,
                        limit=5
                    )
                    if result.matched_contexts:
                        memory_parts = ["[相关记忆参考]"]
                        for ctx in result.matched_contexts:
                            content = ctx.abstract[:300] if len(ctx.abstract) > 300 else ctx.abstract
                            memory_parts.append(f"- {content}")
                        memory_context = "\n".join(memory_parts)
                        logger.info(f"[Memory] Retrieved {len(result.matched_contexts)} contexts")
                except Exception as e:
                    logger.warning(f"[Memory] Failed to retrieve: {e}")

            # Stream response (tool calling is handled internally by chat_service.chat_stream)
            if memory_mode == "v2":
                async for chunk in chat_service.chat_stream(
                    request_with_history, user_preferences, user_id, memory_context, session_service, retriever
                ):
                    full_response.append(chunk)
                    yield f"data: {chunk}\n\n"
            else:
                async for chunk in chat_service.chat_stream(request_with_history, user_preferences, user_id):
                    full_response.append(chunk)
                    yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

            # Save messages based on memory mode
            response_text = "".join(full_response)
            if memory_mode == "v2" and session_service:
                # Use SessionService for v2 mode (handles auto-commit)
                await session_service.add_message(
                    character_id=character_id,
                    topic_id=topic_id,
                    role="user",
                    content=request.message,
                    name=user_id,
                    user_id=user_id
                )
                await session_service.add_message(
                    character_id=character_id,
                    topic_id=topic_id,
                    role="assistant",
                    content=response_text,
                    name=character_name,
                    user_id=user_id
                )
            else:
                # Use history_service for v0/v1 mode
                history_service.append_message(
                    user_id=user_id,
                    topic_id=topic_id,
                    role="user",
                    content=request.message,
                    name=user_id,
                    character_id=character_id
                )
                history_service.append_message(
                    user_id=user_id,
                    topic_id=topic_id,
                    role="assistant",
                    content=response_text,
                    name=character_name,
                    character_id=character_id
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


@router.get("/logs/today")
async def get_today_logs():
    """
    Get today's chat and tool call logs.

    Returns the content of today.txt which contains all logs
    from the current day including tool calls and execution results.
    """
    try:
        from app.utils.file_logger import get_log_content
        from datetime import datetime

        log_content = get_log_content()  # Gets today's logs by default

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
    """
    List all available log files.

    Returns a list of all archived log files with their dates.
    """
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
    """
    Get logs for a specific date.

    Path Parameters:
    - date: Date in YYYY-MM-DD format

    Returns the log content for the specified date.
    """
    try:
        from app.utils.file_logger import get_log_content
        from datetime import datetime

        # Parse date
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        log_content = get_log_content(target_date)

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
