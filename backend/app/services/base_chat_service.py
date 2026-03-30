"""ChatService 抽象基类

统一 v0/v1/v2/v3 的接口，所有版本差异内聚到构造函数。
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

from app.models.character import UserCharacterPreference
from app.schemas.message import ChatRequest, ChatResponse


class BaseChatService(ABC):
    """ChatService 抽象基类，所有版本必须实现此接口。"""

    @abstractmethod
    async def chat(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference] = None,
        user_id: str = "user_default",
    ) -> ChatResponse: ...

    @abstractmethod
    async def chat_stream(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference] = None,
        user_id: str = "user_default",
    ) -> AsyncGenerator[str, None]: ...

    @abstractmethod
    async def persist_messages(
        self,
        character_id: str,
        topic_id: int,
        user_id: str,
        character_name: str,
        user_message: str,
        assistant_message: str,
    ) -> None: ...
