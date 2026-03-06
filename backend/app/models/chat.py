"""Chat history data models for message and topic management."""

import time
import random
import string
from typing import List
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single message in a chat conversation."""
    id: str = Field(..., description="Unique message ID (msg_{timestamp}_{role}_{random})")
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    name: str = Field(..., description="Character/User name")
    content: str = Field(..., description="Message content")
    timestamp: int = Field(..., description="Unix timestamp in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "msg_1770291136799_user_k6mjdfb",
                "role": "user",
                "name": "用户名",
                "content": "你好",
                "timestamp": 1770291136799
            }
        }

    @staticmethod
    def generate_id(role: str) -> str:
        """Generate unique message ID."""
        timestamp_ms = int(time.time() * 1000)
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"msg_{timestamp_ms}_{role}_{random_suffix}"


class ChatTopic(BaseModel):
    """Topic information metadata."""
    topic_id: int = Field(..., description="Topic ID (Unix timestamp)")
    character_id: str = Field(..., description="Character ID")
    created_at: int = Field(..., description="Creation timestamp (from filesystem)")
    updated_at: int = Field(..., description="Last update timestamp (from filesystem)")
    message_count: int = Field(..., description="Number of messages in topic")

    class Config:
        json_schema_extra = {
            "example": {
                "topic_id": 1707523200,
                "character_id": "550e8400-e29b-41d4-a716-446655440000",
                "created_at": 1707523200,
                "updated_at": 1707526800,
                "message_count": 10
            }
        }


class ChatHistory(BaseModel):
    """Chat history containing a list of messages."""
    messages: List[ChatMessage] = Field(default_factory=list, description="List of chat messages")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "id": "msg_1770291136799_user_k6mjdfb",
                        "role": "user",
                        "name": "用户名",
                        "content": "你好",
                        "timestamp": 1770291136799
                    },
                    {
                        "id": "msg_1770291136807_assistant_t5241e8",
                        "role": "assistant",
                        "name": "角色名",
                        "content": "回复内容",
                        "timestamp": 1770291136807
                    }
                ]
            }
        }
