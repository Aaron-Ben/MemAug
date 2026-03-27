# 角色增强记忆系统

一个基于 FastAPI + React 的情感陪伴 AI 系统，提供智能对话、角色定制、记忆管理、技能系统等功能。

## 功能特性

### 💬 智能对话
- **流式响应**：实时 SSE 流式交互体验
- **对话历史**：自动保存和恢复对话上下文
- **V1/V2 双模式**：支持记忆系统切换

### 🎭 角色定制
- 支持自定义 AI 角色性格、行为偏好
- 用户个性化偏好设置
- 自定义提示词系统

### 🧠 记忆系统 (V1/V2)

| 特性 | V1 (日记模式) | V2 (会话模式) |
|------|---------------|---------------|
| 存储 | 文件系统 (diary/) | ChromaDB + 文件 |
| 检索 | VexusIndex 向量检索 | HierarchicalRetriever 层级检索 |
| 记忆提取 | 日记内容提取 | 6类记忆提取 + 去重 |
| 会话压缩 | - | SessionCompressor 长时记忆 |

通过 `MEMORY=v2` 环境变量切换。

### 🎯 Skills 系统 (nanobot 风格)

插件化技能系统，支持动态加载和调用：

- **SkillCreator**: 创建自定义技能
- **Weather**: 天气查询技能

每个 Skill 包含：
- `skill.yaml` - 技能定义
- `main.py` - 技能实现

### 📝 日记系统 (V1)
- AI 自动评估和记录重要对话
- 智能提取主题、标签、感受
- 向量化索引支持语义检索

## 技术栈

### 后端
- **框架**: FastAPI
- **数据库**: SQLite + SQLAlchemy 2.0
- **LLM & Embedding**: OpenRouter (支持多种模型)
- **向量索引**: ChromaDB (V2) / VexusIndex (V1)
- **技能系统**: 自定义 SkillLoader
- **Python**: 3.13+

### 前端
- **框架**: React 18 + TypeScript
- **构建工具**: Vite
- **样式**: Tailwind CSS

## 项目结构

```
emotional-companionship/
├── backend/
│   ├── app/
│   │   ├── api/v1/           # API 路由
│   │   │   ├── chat.py       # 对话接口
│   │   │   ├── character.py # 角色管理
│   │   │   ├── diary.py     # 日记接口
│   │   │   └── skills.py    # 技能接口
│   │   ├── config/          # 配置模块
│   │   ├── models/          # 数据模型
│   │   ├── services/        # 业务逻辑
│   │   └── utils/           # 工具模块
│   ├── memory/
│   │   ├── factory.py       # 记忆工厂 (V1/V2 切换)
│   │   ├── v1/              # V1 日记系统
│   │   └── v2/              # V2 会话系统
│   ├── skills/              # 技能目录
│   │   ├── skill-creator/
│   │   └── weather/
│   ├── plugins/             # 插件系统
│   └── main.py              # 应用入口
├── frontend/                 # React 前端
├── data/                    # 数据目录
│   ├── user/               # V2 用户数据
│   ├── session/            # V2 会话数据
│   ├── diary/              # V1 日记
│   └── logs/               # 日志文件
├── chroma-db/              # ChromaDB 向量存储
├── VectorStore/            # V1 向量索引
└── README.md
```

## 快速开始

### 环境要求
- Python 3.13+
- Node.js 18+
- OpenRouter API Key

### 安装

```bash
# 后端依赖
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 前端依赖
cd frontend && npm install
```

### 配置

复制 `backend/.env.example` 到 `backend/.env`：

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxx
API_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=deepseek/deepseek-v3.2
EmbeddingModel=baai/bge-m3
MEMORY=v2  # 可选: v1 或 v2
```

### 启动

```bash
# 一键启动
make dev

# 或分别启动
# 后端: cd backend && uvicorn main:app --reload
# 前端: cd frontend && npm run dev
```

- 后端: http://localhost:8000
- 前端: http://localhost:5173

## API 文档

启动后访问：http://localhost:8000/docs

### 核心端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/chat/stream` | POST | SSE 流式对话 |
| `/api/v1/character/*` | GET/POST/PATCH/DELETE | 角色管理 |
| `/api/v1/diary/*` | GET/POST | 日记管理 |
| `/api/v1/skills/*` | GET/POST | 技能管理 |
| `/api/v1/chat/logs/*` | GET | 日志查看 |

## Skills 开发

创建新技能：

```
backend/skills/my-skill/
├── skill.yaml     # 技能定义
└── main.py        # 技能实现
```

示例 `skill.yaml`：
```yaml
name: my-skill
description: 我的技能
version: 1.0.0
triggers:
  - "查天气"
actions:
  - name: get_weather
    description: 获取天气
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENROUTER_API_KEY` | API Key | - |
| `OPENROUTER_MODEL` | LLM 模型 | deepseek/deepseek-v3.2 |
| `EmbeddingModel` | Embedding 模型 | baai/bge-m3 |
| `MEMORY` | 记忆系统版本 | v1 |