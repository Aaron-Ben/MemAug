# 情感陪伴 AI 系统

基于 FastAPI + React 的情感陪伴 AI 系统，提供智能对话、角色定制、多代记忆系统（V1/V2/V3）和技能插件等功能。

## 功能特性

### 智能对话
- SSE 流式实时交互
- 自动保存和恢复对话上下文
- 话题（Topic）管理，支持多话题切换

### 角色定制
- 自定义 AI 角色性格、行为偏好
- 用户个性化偏好设置
- 自定义提示词系统

### 记忆系统（V1 / V2 / V3）

| 特性 | V1 日记模式 | V2 会话模式 | V3 知识图谱 |
|------|-------------|-------------|-------------|
| 存储 | 文件系统 + VexusIndex | ChromaDB + 文件 | SQLite (FTS5 + 向量) |
| 检索 | VexusIndex 向量检索 | HierarchicalRetriever 层级检索 | 双路径召回 (精确 + 泛化) |
| 记忆提取 | 日记内容 | 6 类记忆提取 + 去重 | LLM 知识三元组抽取 |
| 图结构 | - | - | 7 种节点 + 7 种边类型 |
| 社区发现 | - | - | Label Propagation + 摘要 |
| 会话压缩 | - | SessionCompressor | EVENT→PATTERN 晋升 |
| 排序算法 | 向量相似度 | 向量相似度 | Personalized PageRank |

通过 `MEMORY` 环境变量选择版本（`v1` / `v2` / `v3`）。

### V3 知识图谱系统

V3 是最新的记忆系统，核心特性：

**节点类型**: USER / PERSON / TOPIC / EVENT / PATTERN / CASE / PREFERENCE

**边类型**: CARES_ABOUT / INVOLVED_IN / TRIGGERS / LEADS_TO / HAS_PREFERENCE / RESOLVED_BY / RELATED_TO

**处理流程**:
1. 每 N 轮对话触发异步知识三元组抽取
2. 双路径召回：精确路径（向量/FTS5 → 社区扩展 → 图游走 → PPR）+ 泛化路径（社区向量 → 成员节点 → 图游走 → PPR）
3. 上下文组装为 XML 注入系统提示词
4. 会话结束时执行 EVENT→PATTERN 晋升与节点失效处理
5. 定期维护：去重 → 全局 PageRank → 社区检测 → 社区摘要

### 技能系统

插件化技能，基于 SKILL.md + YAML frontmatter 定义：

- **Weather**: 天气查询（wttr.in + Open-Meteo）
- 可通过 SkillCreator 创建自定义技能

### 日记系统 (V1)
- AI 自动评估和记录重要对话
- 智能提取主题、标签、感受
- 向量化索引支持语义检索

## 技术栈

### 后端
- **框架**: FastAPI
- **数据库**: SQLite + SQLAlchemy 2.0（V3: SQLite WAL + FTS5）
- **LLM & Embedding**: OpenRouter（支持多种模型）
- **向量索引**: VexusIndex (V1, Rust/PyO3) / ChromaDB (V2) / 内置向量 (V3)
- **图算法**: Personalized PageRank, Label Propagation (V3)
- **Python**: 3.13+

### 前端
- **框架**: React 19 + TypeScript
- **构建工具**: Vite 6
- **样式**: Tailwind CSS
- **路由**: react-router-dom

## 项目结构

```
emotional-companionship/
├── backend/
│   ├── app/
│   │   ├── api/v1/              # API 路由
│   │   │   ├── chat.py          # 对话接口 (含 session/close, graph/stats)
│   │   │   ├── character.py     # 角色管理
│   │   │   ├── chat_history.py  # 话题与历史记录
│   │   │   └── diary.py         # 日记接口 (V1)
│   │   ├── models/              # SQLAlchemy 数据模型
│   │   ├── schemas/             # Pydantic 请求/响应模型
│   │   ├── services/            # 业务逻辑 (分层架构)
│   │   │   ├── base_chat_service.py      # 抽象基类
│   │   │   ├── chat_service_v1.py        # V1 插件式工具调用
│   │   │   ├── chat_service_v2.py        # V2 层级记忆检索
│   │   │   ├── chat_service_v3.py        # V3 知识图谱
│   │   │   ├── character_service.py      # 角色服务
│   │   │   ├── chat_history_service.py   # 话题历史服务
│   │   │   ├── session_service.py        # V2 会话管理
│   │   │   ├── llm.py                    # LLM 调用封装
│   │   │   └── embedding.py              # Embedding 服务
│   │   ├── skills/              # 技能加载器
│   │   │   └── builtin/weather/ # 天气技能
│   │   ├── utils/               # 工具模块
│   │   └── main.py              # 应用入口
│   ├── memory/                  # 记忆系统
│   │   ├── factory.py           # 记忆工厂 (V1/V2/V3 切换)
│   │   ├── v1/                  # V1 日记系统
│   │   │   ├── backend.py       # V1 后端
│   │   │   ├── vector_index.py  # VexusIndex 封装
│   │   │   ├── plugin_manager.py
│   │   │   └── plugins/         # RAG/日记/DeepMemo 插件
│   │   ├── v2/                  # V2 会话系统
│   │   │   ├── backend.py       # V2 后端
│   │   │   ├── retriever.py     # HierarchicalRetriever
│   │   │   ├── memory_extractor.py
│   │   │   ├── memory_deduplicator.py
│   │   │   ├── compressor.py    # SessionCompressor
│   │   │   └── chromadb_manager.py
│   │   └── v3/                  # V3 知识图谱系统
│   │       ├── backend.py       # V3 后端
│   │       ├── config.py        # 图谱配置
│   │       ├── types.py         # 节点/边类型定义
│   │       ├── extractor/       # 知识三元组抽取
│   │       ├── recaller/        # 双路径召回引擎
│   │       ├── format/          # 上下文 XML 组装
│   │       ├── graph/           # 图算法 (PageRank/社区/去重)
│   │       └── store/           # SQLite 存储层
│   ├── plugins/                 # V1 工具调用系统
│   └── .env.example             # 环境变量模板
├── frontend/                    # React 前端
│   └── src/
│       ├── components/          # UI 组件
│       ├── pages/               # 页面组件
│       ├── hooks/               # React Hooks
│       ├── services/            # API 服务层
│       └── contexts/            # React Context
├── vector-db/                   # Rust 向量索引模块 (PyO3)
├── data/                        # 数据目录
│   ├── characters/              # 角色数据
│   ├── diary/                   # V1 日记
│   ├── session/                 # V2 会话数据
│   ├── graph-memory/            # V3 知识图谱 (per-character SQLite)
│   └── logs/                    # 日志文件
├── chroma-db/                   # ChromaDB 向量存储 (V2)
├── VectorStore/                 # V1 向量索引数据
├── Makefile                     # 构建与开发命令
└── README.md
```

## 快速开始

### 环境要求
- Python 3.13+
- Node.js 18+
- Rust toolchain（V1 VexusIndex 模块需要）
- OpenRouter API Key

### 安装

```bash
# 安装全部依赖
make install-backend
make install-frontend
```

### 配置

复制 `backend/.env.example` 到 `backend/.env`：

```env
# 记忆系统版本: v1, v2, v3
MEMORY=v3

OPENROUTER_API_KEY=sk-or-v1-xxxxx
API_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=deepseek/deepseek-v3.2
EmbeddingModel=baai/bge-m3
```

### 启动

```bash
# 一键启动（自动编译 Rust 模块 + 前后端）
make dev

# 或分别启动
make dev-backend   # http://localhost:8000
make dev-frontend  # http://localhost:5173
```

## API 文档

启动后访问：http://localhost:8000/docs

### 核心端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/chat/` | POST | 非流式对话 |
| `/api/v1/chat/stream` | POST | SSE 流式对话 |
| `/api/v1/chat/session/close` | POST | 关闭会话 (V3) |
| `/api/v1/chat/graph/stats` | POST | 图谱统计 (V3) |
| `/api/v1/chat/logs/today` | GET | 今日日志 |
| `/api/v1/chat/logs/list` | GET | 日志列表 |
| `/api/v1/character/*` | CRUD | 角色管理 |
| `/api/v1/chat/topics` | POST/GET/DELETE | 话题管理 |
| `/api/v1/diary/*` | GET/POST/DELETE | 日记管理 (V1) |

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MEMORY` | 记忆系统版本 | `v1` |
| `OPENROUTER_API_KEY` | API Key | - |
| `OPENROUTER_MODEL` | LLM 模型 | `deepseek/deepseek-v3.2` |
| `EmbeddingModel` | Embedding 模型 | `baai/bge-m3` |
| `API_URL` | API 地址 | `https://openrouter.ai/api/v1` |
| `DATABASE_URL` | 数据库 URL | SQLite |
| `DEFAULT_TIMEZONE` | 默认时区 | `Asia/Shanghai` |

## Makefile 命令

| 命令 | 说明 |
|------|------|
| `make dev` | 启动完整开发环境 |
| `make dev-frontend` | 仅启动前端 |
| `make dev-backend` | 仅启动后端 |
| `make build-vector-db` | 编译 Rust 向量索引模块 |
| `make install-frontend` | 安装前端依赖 |
| `make install-backend` | 安装后端依赖 |
| `make clean` | 清理构建文件 |
