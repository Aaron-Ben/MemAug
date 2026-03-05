.PHONY: dev dev-frontend dev-backend build-vector-db install-frontend install-backend clean help

# 默认目标：启动完整开发环境
dev: build-vector-db
	@echo "Starting development environment..."
	@make -j2 dev-frontend dev-backend

# 启动前端开发服务器
dev-frontend:
	@echo "Starting frontend..."
	@cd frontend && npm run dev

# 启动后端开发服务器
dev-backend:
	@echo "Starting backend..."
	@cd backend && source .venv/bin/activate && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 编译 vector-db Python 模块
build-vector-db:
	@echo "Building vector-db Rust module..."
	@cd vector-db && maturin develop --release

# 安装前端依赖
install-frontend:
	@echo "Installing frontend dependencies..."
	@cd frontend && npm install

# 安装后端依赖
install-backend:
	@echo "Installing backend dependencies..."
	@cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 清理构建文件
clean:
	@echo "Cleaning build files..."
	@cd frontend && rm -rf node_modules dist
	@cd backend && rm -rf .venv __pycache__ **/__pycache__
	@cd vector-db && cargo clean

# 帮助信息
help:
	@echo "Available commands:"
	@echo "  make dev              - Start complete development environment (frontend + backend)"
	@echo "  make dev-frontend     - Start frontend only"
	@echo "  make dev-backend      - Start backend only"
	@echo "  make build-vector-db  - Build vector-db Rust module"
	@echo "  make install-frontend - Install frontend dependencies"
	@echo "  make install-backend  - Install backend dependencies"
	@echo "  make clean            - Clean all build files"
	@echo "  make help             - Show this help message"
