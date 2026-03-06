"""Chat history API endpoints for topic and message management."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
import logging

from app.services.chat_history_service import ChatHistoryService
from app.schemas.chat_history import (
    CreateTopicRequest,
    TopicResponse,
    TopicListResponse,
    ChatHistoryResponse,
    ChatMessageResponse,
    DeleteTopicResponse
)
from datetime import datetime

# Create router
router = APIRouter(prefix="/api/v1/chat/topics", tags=["chat-history"])

# Configure logging
logger = logging.getLogger(__name__)


def get_mock_user_id() -> str:
    """Mock user ID for development. In production, this would come from authentication."""
    return "user_default"


def get_chat_history_service() -> ChatHistoryService:
    """Dependency injection for ChatHistoryService."""
    return ChatHistoryService()


@router.post("", response_model=TopicResponse)
async def create_topic(
    request: CreateTopicRequest,
    user_id: str = Depends(get_mock_user_id),
    service: ChatHistoryService = Depends(get_chat_history_service)
):
    """
    Create a new chat topic.

    Request Body:
    - character_id: Character ID (UUID)

    Returns:
        Created topic information with topic_id (Unix timestamp)

    Example:
    ```json
    {
        "character_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    ```
    """
    try:
        character_id = request.character_id

        # Create topic
        topic_id = service.create_topic(user_id, character_id)

        # Get topic info
        topics = service.list_topics(user_id, character_id)
        topic = next((t for t in topics if t.topic_id == topic_id), None)

        if topic is None:
            raise HTTPException(status_code=500, detail="Failed to create topic")

        return TopicResponse(
            topic_id=topic.topic_id,
            character_id=topic.character_id,
            created_at=datetime.fromtimestamp(topic.created_at),
            updated_at=datetime.fromtimestamp(topic.updated_at),
            message_count=topic.message_count
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating topic: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating topic: {str(e)}")


@router.get("", response_model=TopicListResponse)
async def list_topics(
    character_id: Optional[str] = None,
    user_id: str = Depends(get_mock_user_id),
    service: ChatHistoryService = Depends(get_chat_history_service)
):
    """
    List chat topics for a user.

    Query Parameters:
    - character_id: Filter by character ID (UUID)

    Returns:
        List of topics sorted by update time (newest first)

    Example:
    GET /api/v1/chat/topics?character_id=550e8400-e29b-41d4-a716-446655440000
    """
    try:
        # List topics
        topics = service.list_topics(user_id, character_id)

        return TopicListResponse(
            topics=[
                TopicResponse(
                    topic_id=t.topic_id,
                    character_id=t.character_id,
                    created_at=datetime.fromtimestamp(t.created_at),
                    updated_at=datetime.fromtimestamp(t.updated_at),
                    message_count=t.message_count
                )
                for t in topics
            ],
            total=len(topics)
        )

    except Exception as e:
        logger.error(f"Error listing topics: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing topics: {str(e)}")


@router.delete("/{topic_id}", response_model=DeleteTopicResponse)
async def delete_topic(
    topic_id: int,
    character_id: Optional[str] = None,
    user_id: str = Depends(get_mock_user_id),
    service: ChatHistoryService = Depends(get_chat_history_service)
):
    """
    Delete a chat topic.

    Path Parameters:
    - topic_id: Topic ID (Unix timestamp)

    Query Parameters:
    - character_id: Optional character ID for validation

    Returns:
        Success status

    Example:
    DELETE /api/v1/chat/topics/1707523200
    """
    try:
        success = service.delete_topic(user_id, topic_id, character_id)

        if success:
            return DeleteTopicResponse(
                success=True,
                message=f"Topic {topic_id} deleted successfully"
            )
        else:
            return DeleteTopicResponse(
                success=False,
                message=f"Topic {topic_id} not found"
            )

    except Exception as e:
        logger.error(f"Error deleting topic: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting topic: {str(e)}")


@router.get("/{topic_id}/history", response_model=ChatHistoryResponse)
async def get_topic_history(
    topic_id: int,
    character_id: Optional[str] = None,
    user_id: str = Depends(get_mock_user_id),
    service: ChatHistoryService = Depends(get_chat_history_service)
):
    """
    Get chat history for a topic.

    Path Parameters:
    - topic_id: Topic ID (Unix timestamp)

    Query Parameters:
    - character_id: Optional character ID (required if topic not in default location)

    Returns:
        Chat history with messages

    Example:
    GET /api/v1/chat/topics/1707523200/history
    """
    try:
        # Get topic to find character_id if not provided
        if character_id is None:
            topics = service.list_topics(user_id)
            topic = next((t for t in topics if t.topic_id == topic_id), None)
            if topic is None:
                raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found")
            character_id = topic.character_id

        # Get messages
        messages = service.get_topic_history(user_id, topic_id, character_id)

        return ChatHistoryResponse(
            topic_id=topic_id,
            character_id=character_id,
            messages=messages,  # ChatMessage objects already have all fields
            total=len(messages)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting topic history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting topic history: {str(e)}")
