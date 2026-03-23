"""FastAPI application for emotional companionship AI system."""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
env_paths = [
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent.parent / ".env",
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break

# 记忆系统后端选择 (默认 v1)
MEMORY_BACKEND = os.getenv("MEMORY", "v1")
print(f"[Main] Memory backend: {MEMORY_BACKEND}")

# Load environment variables from .env file (try multiple locations)
env_paths = [
    Path(__file__).parent.parent / ".env",  # Backend directory
    Path(__file__).parent.parent.parent / ".env",  # Project root
]
# Try to load from backend directory first, then project root
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break

from app.api.v1 import character, chat, diary, chat_history
from app.services.character_service import CharacterStorageService
from app.models.database import init_db
from app.utils.file_logger import DailyFileHandler, LOGS_DIR

# Configure logging for both application and uvicorn
# Setup root logger to ensure uvicorn access logs work
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Console handler for all logs (including uvicorn access logs)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
root_logger.addHandler(console_handler)

# Create shared file handler for all tool-related logs
file_handler = DailyFileHandler()
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

# Add file handler to loggers
for logger_name in ["app.services.chat_service_v0", "app.services.chat_service_v1", "plugins.tool_executor", "plugins.plugin", "plugins.tool_call_parser", "app.api.v1.chat", "app.api.v1.diary", "app.vector_index"]:
    logger = logging.getLogger(logger_name)
    logger.addHandler(file_handler)
    # Also add console handler so logs appear in terminal
    logger.addHandler(console_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger = logging.getLogger(__name__)

    # 启动时初始化
    init_db()
    print("Database initialized successfully")

    # ✅ 先初始化 VectorIndex，再加载插件（修复初始化顺序问题）
    from app.vector_index import initialize_vector_index
    from memory.v1.plugin_manager import plugin_manager

    try:
        vector_index = await initialize_vector_index()
        plugin_manager.set_vector_db_manager(vector_index)
        print("✅ VectorIndex 已初始化并注入到 plugin_manager")
    except Exception as e:
        print(f"❌ VectorIndex 初始化失败: {e}")

    try:
        await plugin_manager.load_plugins()
        print(f"Plugins loaded: {list(plugin_manager.plugins.keys())}")
    except Exception as e:
        print(f"Failed to load plugins: {e}")

    # Ensure logs directory exists
    from app.utils.file_logger import LOGS_DIR
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"File logging enabled: {LOGS_DIR}/today.txt")

    # 启动时自动同步所有角色的日记到向量索引
    try:
        from app.vector_index import sync_all_diaries_to_vector_index
        logger.info("🚀 启动时自动同步向量索引...")
        await sync_all_diaries_to_vector_index()
    except Exception as e:
        logger.error(f"❌ 启动时向量索引同步失败: {e}", exc_info=True)

    yield

    # Close file loggers
    for handler in list(logging.getLogger().handlers):
        if isinstance(handler, DailyFileHandler):
            handler.close()

    print("Application shutting down")


# Create FastAPI app
app = FastAPI(
    title="Emotional Companionship AI",
    description="An AI-powered emotional companionship system with customizable characters",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize character service
character_service = CharacterStorageService()

# Store services in app state for dependency injection
app.state.character_service = character_service


# Include routers
app.include_router(character.router)
app.include_router(chat.router)
app.include_router(diary.router)
app.include_router(chat_history.router)


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Emotional Companionship AI API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "characters": "/api/v1/character/",
            "chat": "/api/v1/chat/",
            "diary": "/api/v1/diary/",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "characters_loaded": len(character_service.list_characters())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
