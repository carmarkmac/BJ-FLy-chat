# 启动配置文档

## 1. 项目路径

- 项目根目录：克隆后的仓库根目录（下文记为 `<项目根>`）
- 服务目录：`<项目根>/serve`

## 2. 运行环境

- 操作系统：Windows、macOS、Linux
- Python：推荐 `3.9`（项目依赖更兼容）
- 包管理：`pip` / `conda`

建议环境：

```bash
conda create -n llm-universe python=3.9 -y
conda activate llm-universe
```

## 3. 关键依赖

若出现缺包，可安装以下核心依赖：

```bash
pip install "langchain==0.0.292" "langsmith==0.0.92" "openai>=1.12.0,<2" fastapi uvicorn gradio python-dotenv zhipuai ipython
```

`get_completion_gpt`（含 Deepseek `deepseek-chat`）已使用 **OpenAI Python SDK 1.x** 的 `OpenAI` 客户端，需 `openai>=1.0`，勿再降级到 `0.28`。

### 3.1 Deepseek（OpenAI 兼容）

在 `.env` 中设置：

- `OPENAI_API_KEY`：Deepseek 的 API Key  
- `OPENAI_API_BASE`：例如 `https://api.deepseek.com/v1`（以官方文档为准）

## 4. 启动命令（前后端）

在两个终端分别执行：

### 4.1 启动后端 API（FastAPI）

```bash
cd <项目根>/serve
python -m uvicorn api:app --host 0.0.0.0 --port 8001
```

访问地址：

- 本机：`http://127.0.0.1:8001`
- 局域网：`http://<你的机器IP>:8001`

> 说明：默认 `8000` 端口可能被占用，当前使用 `8001`。

### 4.2 启动前端（Gradio）

```bash
cd <项目根>/serve
python run_gradio.py
```

访问地址：

- 本机：`http://127.0.0.1:7860`

## 5. 端口占用排查

查看端口监听：

```bash
lsof -nP -iTCP:8001 -sTCP:LISTEN
lsof -nP -iTCP:7860 -sTCP:LISTEN
```

如需释放端口（替换 `<PID>`）：

```bash
kill <PID>
```

## 6. 快速自检

- 打开前端地址 `http://127.0.0.1:7860`，页面可访问
- 后端启动日志包含：`Uvicorn running on http://0.0.0.0:8001`
- 前端提交请求时无缺包报错（如 `ModuleNotFoundError`）

## 7. 常见问题

- `ModuleNotFoundError: No module named 'langchain.prompts'`
  - 原因：`langchain` 版本过高（1.x）与项目代码不兼容
  - 处理：安装 `langchain==0.0.292`

- `ModuleNotFoundError: No module named 'IPython'`
  - 处理：`pip install ipython`

- `Address already in use`
  - 处理：更换端口或先释放占用进程

- `You tried to access openai.ChatCompletion...`
  - 原因：旧代码与 `openai>=1.0` 不兼容（已改为 `client.chat.completions.create`）
  - 处理：使用 `openai>=1.12.0,<2` 并拉取最新代码；若仍报错可执行 `pip install -U "openai>=1.12.0,<2"`
