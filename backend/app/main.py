"""FastAPI application for emotional companionship AI system."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path

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
for logger_name in ["app.services.chat_service", "plugins.tool_executor", "plugins.plugin", "plugins.tool_call_parser", "app.api.v1.chat", "app.api.v1.diary", "app.vector_index"]:
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

    # 检查 Genie-TTS 资源目录
    from pathlib import Path
    from app.characters.tts import GENIE_DATA_DIR, load_predefined_character

    genie_data_path = Path(GENIE_DATA_DIR) if GENIE_DATA_DIR else None
    if genie_data_path and genie_data_path.exists():
        print(f"Genie-TTS data directory found at {genie_data_path}")
    else:
        print("Genie-TTS data directory not found, will download on first use")

    # 预加载默认角色
    try:
        load_predefined_character("feibi")  # 菲比 - 中文角色
        print("Genie-TTS character 'feibi' loaded successfully")
    except Exception as e:
        print(f"Failed to load Genie-TTS character: {e}")

    # Load plugins
    from plugins.plugin import plugin_manager
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
    # 关闭时的清理工作（如果需要）
    from app.characters.tts import cleanup
    cleanup()

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
