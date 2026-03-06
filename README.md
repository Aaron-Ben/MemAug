# 情感陪伴 AI 系统

一个基于 FastAPI + React 的情感陪伴 AI 系统，提供智能对话、角色定制、日记记录、工具调用等功能。

## 功能特性

### 💬 智能对话
- **流式响应**：实时交互体验
- **语音输入**：支持中英日韩粤语识别
- **语音合成**：TTS 语音输出
- **对话历史**：自动保存和恢复对话上下文

### 🎭 角色定制
- 支持自定义 AI 角色性格、行为偏好
- 用户个性化偏好设置
- 自定义提示词系统

### 🔧 工具调用系统 (VCPToolBox 模式)
- **插件化架构**：模块化扩展功能，支持多种协议
  - **stdio 协议**：通过子进程执行外部插件（如 DeepMemo Rust 插件）
  - **direct 协议**：直接调用 Python 插件（如 RAG 日记检索）
- **自动工具检测**：AI 响应中的工具调用标记自动检测和执行
- **工具调用格式**：`<<<[TOOL_REQUEST]>>>...<<<[END_TOOL_REQUEST]>>>`
- **工具结果注入**：执行结果自动注入下一轮对话

### 内置插件
1. **DeepMemo（深度回忆）**
   - 基于向量数据库的语义检索
   - 支持时间范围查询（如"三天前"、"上周"）
   - 智能标签关联和上下文理解

2. **RAGDailyPlugin（日记检索）**
   - 基于用户日记的 RAG 检索
   - 支持时间表达式解析
   - 自动分块处理长文本

3. **DailyNote（日记管理）**
   - AI 自动创建和更新日记
   - 支持智能内容提取
   - 自动标签生成

### 📝 智能日记
- **AI 自动触发**：每次对话后 AI 自动评估是否值得记录
  - 智能判断对话重要性
  - 只记录有意义的对话内容
- **智能提取**：
  - 从对话中提取结构化日记内容
  - 自动生成对话主题、关键信息、感受
  - 自动添加相关标签
- **日记管理**：
  - 基于文件系统存储
  - 支持创建、更新、删除日记
  - AI 可以主动修正和补充已有日记
- **向量索引系统**：
  - 自动将日记文件分块并向量化
  - 支持高效语义检索和相似度搜索
  - 懒加载索引优化内存使用
  - 延迟保存策略减少磁盘 I/O
- **记忆集成**：
  - 后续对话自动参考相关日记内容
  - AI 能够记住与用户的重要时刻

### 📊 日志系统
- **文件日志记录**：所有对话和工具调用自动记录到文件
- **按日期归档**：
  - 当天日志：`data/logs/today.txt`
  - 历史日志：`data/logs/YYYY-MM-DD.txt`
- **实时写入**：日志实时刷新，方便查看
- **日志 API**：
  - `GET /api/v1/chat/logs/today` - 获取今日日志
  - `GET /api/v1/chat/logs/list` - 列出所有日志文件
  - `GET /api/v1/chat/logs/{date}` - 获取指定日期日志

### 🎨 用户界面
- React + TypeScript 实现
- 响应式布局，支持移动端
- 实时消息流式显示

## 技术栈

### 后端
- **框架**: FastAPI
- **数据库**: SQLite + SQLAlchemy 2.0
- **LLM & Embedding**: OpenRouter (支持 BGE-M3 等模型)
- **向量索引**: VexusIndex (自研向量数据库)
- **插件系统**: 自定义插件管理器，支持 stdio/direct 协议
- **语音识别**: Sherpa-ONNX SenseVoice
- **语音合成**: Genie-TTS (GPT-SoVITS)
- **Python**: 3.13+

### 前端
- **框架**: React 18 + TypeScript
- **构建工具**: Vite
- **样式**: Tailwind CSS
- **状态管理**: React Hooks

## 项目结构

```
emotional-companionship/
├── backend/                      # 后端服务
│   ├── app/
│   │   ├── api/               # API 路由
│   │   │   └── v1/
│   │   │       ├── chat.py          # 对话接口（含日志端点）
│   │   │       ├── character.py     # 角色管理
│   │   │       ├── diary.py         # 日记接口
│   │   │       └── chat_history.py  # 对话历史接口
│   │   ├── config/            # 配置模块
│   │   ├── models/            # 数据模型
│   │   │   ├── database.py    # 数据库模型（日记、向量存储）
│   │   │   └── ...
│   │   ├── schemas/           # Pydantic 模型
│   │   ├── services/          # 业务逻辑
│   │   │   ├── chat_service.py         # 对话服务（含工具调用）
│   │   │   ├── character_service.py    # 角色服务
│   │   │   ├── llm.py                   # LLM 服务
│   │   │   ├── chat_history_service.py  # 对话历史服务
│   │   │   ├── chunk_text.py            # 文本分块服务
│   │   │   ├── embedding.py             # 向量化服务
│   │   │   └── diary/                   # 日记服务
│   │   │       └── file_service.py
│   │   ├── utils/            # 工具模块
│   │   │   ├── file_logger.py         # 文件日志记录
│   │   │   └── json.py               # JSON 工具
│   │   ├── characters/       # 角色模块
│   │   │   ├── asr.py          # 语音识别
│   │   │   └── tts.py          # 语音合成
│   │   └── vector_index.py  # 向量索引系统
│   ├── plugins/           # 插件系统
│   │   ├── plugin.py              # 插件管理器
│   │   ├── tool_call_parser.py   # 工具调用解析器
│   │   ├── tool_executor.py      # 工具执行器
│   │   ├── daily_note/           # DailyNote 插件
│   │   ├── deepmemo/             # DeepMemo 插件（Rust）
│   │   └── rag_daily/            # RAG 日记检索插件
│   ├── tests/             # 测试目录
│   │   └── test_vector_index.py  # 向量索引测试
│   ├── resources/        # 资源文件
│   └── main.py           # 应用入口
│
├── frontend/                  # 前端应用
│   └── src/
│       ├── components/      # React 组件
│       ├── pages/         # 页面组件
│       ├── hooks/         # 自定义 Hooks
│       ├── services/       # API 服务
│       └── types/         # TypeScript 类型
│
├── data/                  # 数据目录
│   ├── chat/           # 对话历史
│   ├── diary/          # 日记存储
│   ├── logs/           # 日志文件（today.txt + YYYY-MM-DD.txt）
│   └── characters/     # 角色数据（日记、配置等）
├── VectorStore/         # 向量索引存储目录
│
├── Makefile             # 开发命令
└── README.md            # 项目文档
```

## 快速开始

### 环境要求
- Python 3.13+
- Node.js 18+
- Rust 工具链（用于编译 vector-db）
- OpenRouter API Key

### 安装依赖

**后端依赖：**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**前端依赖：**
```bash
make install-frontend
# 或
cd frontend && npm install
```

**编译 Rust 模块：**
```bash
make build-vector-db
# 或
cd vector-db && maturin develop --release
```

### 配置环境变量

复制 `backend/.env.example` 到 `backend/.env` 并填入你的 API Key：

```env
# OpenRouter API Key (必需)
OPENROUTER_API_KEY=sk-or-v1-xxxxx

# API 配置
API_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=deepseek/deepseek-v3.2
EmbeddingModel=baai/bge-m3
```

### 初始化数据库

```bash
cd backend
python -c "from app.models.database import init_db; init_db()"
```

### 启动项目

**一键启动（推荐）：**
```bash
make dev
```

这将自动启动前端和后端服务。


后端将运行在 `http://localhost:8000`
前端将运行在 `http://localhost:5173`

## API 文档

启动后端后，访问以下地址查看 API 文档：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 核心 API 端点

#### 对话
- `POST /api/v1/chat/` - 发送消息
- `POST /api/v1/chat/stream` - SSE 流式对话
- `POST /api/v1/chat/voice` - 语音识别
- `POST /api/v1/chat/tts` - 文字转语音
- `GET /api/v1/chat/tts/audio/{filename}` - 获取生成的语音文件
- `GET /api/v1/chat/logs/today` - 获取今日日志
- `GET /api/v1/chat/logs/list` - 列出所有日志文件
- `GET /api/v1/chat/logs/{date}` - 获取指定日期日志

#### 角色
- `GET /api/v1/character/user/list` - 获取角色列表
- `GET /api/v1/character/user/{character_id}` - 获取角色详情
- `POST /api/v1/character/create` - 创建新角色
- `PATCH /api/v1/character/user/{character_id}` - 更新角色提示词
- `DELETE /api/v1/character/user/{character_id}` - 删除角色

#### 日记
- `GET /api/v1/diary/list` - 获取日记列表
- `GET /api/v1/diary/latest` - 获取最新日记
- `GET /api/v1/diary/names` - 获取日记本名称列表
- `GET /api/v1/diary/sync` - 同步日记到数据库
- `GET /api/v1/diary/{path:path}` - 获取指定日记
- `POST /api/v1/diary/create` - 创建日记
- `POST /api/v1/diary/ai-update` - AI 更新日记（查找替换）
- `DELETE /api/v1/diary/{path:path}` - 删除日记

#### 对话历史
- `POST /api/v1/chat/topics` - 创建新的对话主题
- `GET /api/v1/chat/topics` - 列出对话主题
- `GET /api/v1/chat/topics/{topic_id}/history` - 获取对话历史
- `DELETE /api/v1/chat/topics/{topic_id}` - 删除对话主题

## 工具调用开发指南

### 工具调用格式

AI 需要使用工具时，按以下格式输出：

```
<<<[TOOL_REQUEST]>>>
tool_name:「始」插件名「末」,
参数1:「始」值1「末」,
参数2:「始」值2「末」
<<<[END_TOOL_REQUEST]>>>
```

### 插件清单

当前可用的插件：

| 插件名 | 协议 | 功能 | 配置 |
|---------|--------|------|------|
| DeepMemo | stdio | 向量语义检索 | `plugins/deepmemo/config.env` |
| RAGDailyPlugin | direct | 日记 RAG 检索 | `plugins/rag_daily/config.env` |
| DailyNote | direct | 日记创建和更新 | - |

### 创建新插件

1. **Stdio 协议插件**（如 Rust/Go 实现）：
   - 创建插件目录：`plugins/your_plugin/`
   - 创建 `plugin-manifest.json`：
     ```json
     {
       "name": "YourPlugin",
       "displayName": "你的插件",
       "version": "1.0.0",
       "communication": {
         "protocol": "stdio",
         "timeout": 60000
       },
       "entryPoint": {
         "command": "./target/release/your_plugin"
       }
     }
     ```
   - 可选：创建 `config.env` 配置文件

2. **Direct 协议插件**（Python 实现）：
   - 创建插件目录：`plugins/your_plugin/`
   - 创建 `plugin-manifest.json`：
     ```json
     {
       "name": "YourPlugin",
       "displayName": "你的插件",
       "version": "1.0.0",
       "communication": {
         "protocol": "direct"
       },
       "entryPoint": {
         "script": "main.py"
       },
       "capabilities": {
         "invocationCommands": [
           {
             "command": "ToolName",
             "description": "工具描述",
             "example": "{\"param\": \"value\"}"
           }
         ]
       }
     }
     ```
   - 实现 `main.py`：
     ```python
     async def initialize(config, dependencies):
         # 初始化逻辑
         pass

     async def process_tool_call(args):
         # 工具调用处理
         return {"status": "success", "result": "..."}
     ```

## 配置说明

### 向量索引系统

系统内置向量索引功能，用于高效存储和检索日记内容：

**核心特性：**
- 自动文本分块（基于 token 限制）
- 批量向量化处理
- 懒加载索引（按需加载）
- 延迟保存策略（减少磁盘 I/O）
- 支持增量更新

**使用方法：**
```python
from app.vector_index import VectorIndex, VectorIndexConfig

# 创建索引实例
config = VectorIndexConfig()
vector_index = VectorIndex(config)

# 处理单个日记文件
result = await vector_index.process_diary_file(
    character_id="sister_001",
    file_path="2024-01-15.txt"
)

# 批量同步角色的所有日记
result = await vector_index.sync_character_diaries("sister_001")

# 保存所有索引到磁盘
await vector_index.flush_all()
```

**索引文件位置：**
- 存储目录：`VectorStore/`
- 文件命名：`index_diary_{MD5}.usearch`
- 每个角色有独立的索引文件

**运行测试：**
```bash
cd backend
python tests/test_vector_index.py
```

### 环境变量 (.env)

| 变量 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `OPENROUTER_API_KEY` | OpenRouter API Key (LLM + Embedding) | - | ✅ |
| `API_URL` | OpenRouter API 地址 | `https://openrouter.ai/api/v1` | ✅ |
| `OPENROUTER_MODEL` | LLM 模型名称 | `anthropic/claude-3.5-sonnet` | ✅ |
| `EmbeddingModel` | Embedding 模型名称 | `baai/bge-m3` | ✅ |

## 日志说明

日志文件位于 `data/logs/` 目录：

- **today.txt**：当天的日志（实时追加）
- **YYYY-MM-DD.txt**：历史日志（自动归档）

日志内容包括：
- 对话信息
- 工具调用检测和执行
- 插件执行结果
- 错误和异常信息

查看日志示例：
```bash
# 查看今天的日志
curl http://localhost:8000/api/v1/chat/logs/today

# 列出所有日志
curl http://localhost:8000/api/v1/chat/logs/list
```