# BJ-FLy-chat

基于 LangChain 的 Datawhale 知识库问答助手，支持多种大语言模型，提供 Gradio 聊天界面与 FastAPI 后端。

## 功能特性

- 基于 Datawhale 开源项目 README 的知识检索与问答
- 支持 OpenAI、文心一言、讯飞星火、智谱 GLM 等模型
- Gradio Web 界面 + FastAPI 服务
- 本地向量库（Chroma），支持 m3e 等 Embedding 模型

## 环境要求

- Python >= 3.9
- 内存 >= 4 GB
- Windows / macOS / Linux

## 快速开始

### 1. 克隆项目

```shell
git clone https://github.com/carmarkmac/BJ-FLy-chat.git
cd BJ-FLy-chat
```

### 2. 安装依赖

```shell
conda create -n bj-fly python=3.9
conda activate bj-fly
pip install -r requirements.txt
```

### 3. 配置环境变量

```shell
cp .env.example .env
```

编辑 `.env`，填入所用模型的 API Key 及服务端口等配置，详见 `.env.example` 中的注释。

### 4. 启动服务

**一键启动（推荐）：**

```shell
chmod +x restart_services.sh
./restart_services.sh
```

**手动启动：**

```shell
# 后端 API（默认端口 8001）
cd serve
uvicorn api:app --host 0.0.0.0 --port 8001

# Gradio 前端（默认端口 7860，另开终端）
cd serve
python run_gradio.py
```

启动后访问 `http://127.0.0.1:7860` 使用问答界面。

## 项目结构

```
├── database/       # 知识库构建与向量库管理
├── embedding/      # Embedding 模型封装
├── knowledge_db/   # 知识库源文件
├── llm/            # 大模型 API 封装
├── qa_chain/       # 检索问答链
├── serve/          # Gradio 与 FastAPI 服务入口
├── vector_db/      # 向量库持久化目录（运行后生成）
└── doc/            # 运维与部署文档
```

## 支持的模型

| 厂商 | 模型示例 |
|------|----------|
| OpenAI | gpt-3.5-turbo、gpt-4 等 |
| 文心一言 | ERNIE-Bot、ERNIE-Bot-4、ERNIE-Bot-turbo |
| 讯飞星火 | Spark-1.5、Spark-2.0 |
| 智谱 AI | chatglm_pro、chatglm_std、chatglm_lite |

## 知识库维护

- 知识库源文件位于 `knowledge_db/` 目录
- 可通过 `database/create_db.py` 重建向量库
- 更多说明见 `doc/SOP-03-知识库维护.md`
