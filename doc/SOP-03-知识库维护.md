# SOP-03 知识库维护

## 1. 知识库文件位置

```
knowledge_db/        ← 原始文档（.md / .txt / .pdf）
vector_db/chroma/    ← 向量化后的持久化索引
```

---

## 2. 添加新文档

1. 将新文件放入 `knowledge_db/` 目录。
2. 重建向量库（见第 3 节）。

支持格式：`.md`、`.txt`、`.pdf`（需安装 `pypdf`）。

---

## 3. 重建向量库

知识库文件有增删时，需重建向量索引：

```bash
conda activate bj-fly
cd <项目根>
python database/create_db.py
```

重建完成后重启 Gradio 服务使新索引生效。

> 注意：重建会覆盖 `vector_db/chroma/` 下的旧索引，操作前可先备份该目录。

---

## 4. 知识盲区排查

查询知识库无法回答的问题（`is_unanswered = 1`）：

```bash
sqlite3 qa_log.db \
  "SELECT question, created_at FROM questions WHERE is_unanswered = 1 ORDER BY created_at DESC LIMIT 50;"
```

根据高频未回答问题，补充对应文档到 `knowledge_db/`，然后重建向量库。

---

## 5. 验证知识库加载

启动 Gradio 后，在日志中确认以下输出：

```
[知识库问答-带记忆] 知识库路径: .../knowledge_db
[知识库问答-带记忆] 向量库路径: .../vector_db/chroma
[知识库问答-带记忆] 问答链创建成功
```

若出现 `FileNotFoundError`，检查路径是否正确，或重新执行 `create_db.py`。
