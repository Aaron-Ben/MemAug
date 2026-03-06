"""Enhanced message schemas with character context support."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class MessageContext(BaseModel):
    """Enhanced context for character-aware messages."""
    recent_conversation_summary: Optional[str] = Field(None, description="Summary of recent conversation")
    character_state: Dict[str, Any] = Field(default_factory=dict, description="Character's current internal state")
    initiate_topic: bool = Field(default=False, description="Whether character should initiate a topic")



class ChatRequest(BaseModel):
    """Request for character chat."""
    message: str = Field(..., description="User's message to the character")
    character_id: str = Field(default="sister_001", description="Character to chat with")
    topic_id: Optional[int] = Field(None, description="Topic ID for continuing a conversation")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, description="Previous conversation messages for context"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "我回来了",
                "character_id": "550e8400-e29b-41d4-a716-446655440000",
                "conversation_history": [
                    {"role": "user", "content": "我出门了"},
                    {"role": "assistant", "content": "哥哥路上小心～"}
                ],
                "stream": False
            }
        }


class ChatResponse(BaseModel):
    """Response from character chat."""
    message: str = Field(..., description="Character's response message")
    character_id: str = Field(..., description="Character that generated the response")
    topic_id: Optional[int] = Field(None, description="Topic ID for the conversation")
    context_used: Optional[MessageContext] = Field(None, description="Context information used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "哥哥回来啦！今天过得怎么样呀？我等你等好久呢～",
                "character_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


class StreamChatResponse(BaseModel):
    """Chunk of a streaming chat response."""
    chunk: str = Field(..., description="Chunk of the response text")
    character_id: str = Field(..., description="Character generating the response")
    done: bool = Field(default=False, description="Whether this is the final chunk")


class VoiceResponse(BaseModel):
    """Response from voice input processing."""
    text: str = Field(..., description="Recognized text from voice input")
    emotion: Optional[str] = Field(None, description="Detected emotion from voice (e.g., [开心], [伤心])")
    event: Optional[str] = Field(None, description="Detected event from voice (e.g., [鼓掌], [大笑])")
    success: bool = Field(..., description="Whether recognition was successful")
    error: Optional[str] = Field(None, description="Error message if recognition failed")


class TTSRequest(BaseModel):
    """Request for text-to-speech synthesis."""
    text: str = Field(..., description="Text to synthesize to speech")
    engine: str = Field(default="vits", description="TTS engine: 'vits' or 'pyttsx3'")
    character_id: str = Field(default="sister_001", description="Character ID (for voice selection)")



class TTSResponse(BaseModel):
    """Response from text-to-speech synthesis."""
    success: bool = Field(..., description="Whether synthesis was successful")
    audio_path: Optional[str] = Field(None, description="Path to the generated audio file")
    error: Optional[str] = Field(None, description="Error message if synthesis failed")
    engine: str = Field(..., description="TTS engine used")
