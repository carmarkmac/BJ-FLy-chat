import sys
import os
import traceback
import logging
import threading
import time
import re
import inspect

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("Step 1: 开始导入 gradio...")
import gradio as gr
import gradio_client.utils as gradio_client_utils
logger.info("Step 2: 开始导入环境变量...")
from dotenv import load_dotenv, find_dotenv
logger.info("Step 3: 开始导入 LLM 模块...")
from llm.call_llm import get_completion
logger.info("Step 4: 开始导入 QA chain 模块...")
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
from qa_chain.QA_chain_self import QA_chain_self
from database.qa_log import (
    save_qa, register_user, login_user,
    save_chat_turn, load_chat_history, get_next_turn_index,
    generate_invite_code, SessionLocal, InviteCode,
    submit_ip_change,
)
from starlette.templating import Jinja2Templates
logger.info("Step 5: 所有模块导入完成")


def patch_gradio_client_schema_bool():
    if getattr(gradio_client_utils, "_schema_bool_patched", False):
        return
    original_converter = gradio_client_utils._json_schema_to_python_type

    def json_schema_to_python_type_compat(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return original_converter(schema, defs)

    gradio_client_utils._json_schema_to_python_type = json_schema_to_python_type_compat
    gradio_client_utils._schema_bool_patched = True


def patch_starlette_template_response():
    signature = inspect.signature(Jinja2Templates.TemplateResponse)
    params = list(signature.parameters.values())
    if len(params) < 2 or params[1].name != "request":
        return

    original_template_response = Jinja2Templates.TemplateResponse

    def template_response_compat(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.pop("context", {}) or {}
            request = context.get("request")
            if request is None:
                raise ValueError("Template context must include request")
            return original_template_response(
                self, request=request, name=name, context=context,
                status_code=kwargs.get("status_code", 200),
                headers=kwargs.get("headers"),
                media_type=kwargs.get("media_type"),
                background=kwargs.get("background"),
            )
        return original_template_response(self, *args, **kwargs)

    Jinja2Templates.TemplateResponse = template_response_compat


patch_gradio_client_schema_bool()
patch_starlette_template_response()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env_path = os.path.join(_PROJECT_ROOT, ".env")
logger.info(f"项目根目录: {_PROJECT_ROOT}")
logger.info(f".env 路径: {_env_path}")
logger.info(f".env 文件存在: {os.path.isfile(_env_path)}")
_ = load_dotenv(_env_path) if os.path.isfile(_env_path) else load_dotenv(find_dotenv())

api_key = os.environ.get("OPENAI_API_KEY", "")
api_base = os.environ.get("OPENAI_API_BASE", "")
logger.info(f"OPENAI_API_KEY 已设置: {bool(api_key and len(api_key) > 10)}")
logger.info(f"OPENAI_API_BASE: {api_base}")

if not api_key or len(api_key) < 10:
    logger.error("❌ OPENAI_API_KEY 未设置或格式错误！")
if not api_base:
    logger.warning("⚠️  OPENAI_API_BASE 未设置，将使用默认 OpenAI 地址")
elif "deepseek" not in api_base.lower():
    logger.warning(f"⚠️  OPENAI_API_BASE 不是 DeepSeek 地址: {api_base}")

LLM_MODEL_DICT = {
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"],
    "wenxin": ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"],
    "xinhuo": ["Spark-1.5", "Spark-2.0"],
    "zhipuai": ["chatglm_pro", "chatglm_std", "chatglm_lite"]
}

LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()), [])
INIT_LLM = "deepseek-chat"
EMBEDDING_MODEL_LIST = ['zhipuai', 'openai', 'm3e']
INIT_EMBEDDING_MODEL = "m3e"
DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "knowledge_db")
DEFAULT_PERSIST_PATH = os.path.join(_PROJECT_ROOT, "vector_db", "chroma")


def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform, "")


class Model_center():
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}
        self.vectordb_cache = {}  # 缓存向量数据库

    def chat_qa_chain_self_answer(self, question, chat_history=[], model="openai", embedding="openai",
                                  temperature=0.0, top_k=4, history_len=3,
                                  file_path=DEFAULT_DB_PATH, persist_path=DEFAULT_PERSIST_PATH):
        logger.info(f"[知识库问答-带记忆] 收到问题: {question[:50]}...")
        if question is None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.chat_qa_chain_self:
                # 先检查是否有缓存的向量数据库
                vdb_key = (embedding, persist_path)
                if vdb_key not in self.vectordb_cache:
                    logger.info(f"[知识库问答-带记忆] 首次加载向量数据库: {persist_path}")
                    from qa_chain.get_vectordb import get_vectordb
                    self.vectordb_cache[vdb_key] = get_vectordb(file_path, persist_path, embedding, None)
                    logger.info(f"[知识库问答-带记忆] 向量数据库缓存完成")
                else:
                    logger.info(f"[知识库问答-带记忆] 使用缓存的向量数据库")
                
                self.chat_qa_chain_self[(model, embedding)] = Chat_QA_chain_self(
                    model=model, temperature=temperature, top_k=top_k,
                    chat_history=chat_history, file_path=file_path,
                    persist_path=persist_path, embedding=embedding,
                    vectordb=self.vectordb_cache[vdb_key])
            chain = self.chat_qa_chain_self[(model, embedding)]
            result = chain.answer(question=question, temperature=temperature, top_k=top_k)
            return "", result
        except Exception as e:
            logger.error(f"[知识库问答-带记忆] ❌ {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return "抱歉，系统处理您的问题时出现了错误，请稍后重试。", chat_history

    def qa_chain_self_answer(self, question, chat_history=[], model="openai", embedding="openai",
                             temperature=0.0, top_k=4,
                             file_path=DEFAULT_DB_PATH, persist_path=DEFAULT_PERSIST_PATH):
        logger.info(f"[知识库问答-无记忆] 收到问题: {question[:50]}...")
        if question is None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.qa_chain_self:
                # 先检查是否有缓存的向量数据库
                vdb_key = (embedding, persist_path)
                if vdb_key not in self.vectordb_cache:
                    logger.info(f"[知识库问答-无记忆] 首次加载向量数据库: {persist_path}")
                    from qa_chain.get_vectordb import get_vectordb
                    self.vectordb_cache[vdb_key] = get_vectordb(file_path, persist_path, embedding, None)
                    logger.info(f"[知识库问答-无记忆] 向量数据库缓存完成")
                else:
                    logger.info(f"[知识库问答-无记忆] 使用缓存的向量数据库")
                
                self.qa_chain_self[(model, embedding)] = QA_chain_self(
                    model=model, temperature=temperature, top_k=top_k,
                    file_path=file_path, persist_path=persist_path, embedding=embedding,
                    vectordb=self.vectordb_cache[vdb_key])
            chain = self.qa_chain_self[(model, embedding)]
            answer = chain.answer(question, temperature, top_k)
            chat_history.append((question, answer))
            return "", chat_history
        except Exception as e:
            logger.error(f"[知识库问答-无记忆] ❌ {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return "抱歉，系统处理您的问题时出现了错误，请稍后重试。", chat_history

    def clear_history(self):
        for chain in self.chat_qa_chain_self.values():
            chain.clear_history()


def format_chat_prompt(message, chat_history):
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt


def respond(message, chat_history, llm, history_len=3, temperature=0.1, max_tokens=2048):
    logger.info(f"[直接对话] 收到消息: {message[:50]}...")
    if message is None or len(message) < 1:
        return "", chat_history
    try:
        chat_history = chat_history[-history_len:] if history_len > 0 else []
        formatted_prompt = format_chat_prompt(message, chat_history)
        bot_message = get_completion(formatted_prompt, llm, temperature=temperature, max_tokens=max_tokens)
        bot_message = re.sub(r"\\n", '<br/>', bot_message)
        chat_history.append((message, bot_message))
        return "", chat_history
    except Exception as e:
        logger.error(f"[直接对话] ❌ {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return "抱歉，系统处理您的问题时出现了错误，请稍后重试。", chat_history


model_center = Model_center()

# --- Seed first invite code if none exist ---
try:
    with SessionLocal() as _s:
        if _s.query(InviteCode).count() == 0:
            _code = generate_invite_code()
            logger.info(f"[首次启动] 初始邀请码（请妥善保管）: {_code}")
except Exception:
    pass

# --- Online user tracking ---
_online_users: dict = {}
_users_lock = threading.Lock()
_SESSION_TIMEOUT = 120


def _cleanup_stale_sessions():
    while True:
        time.sleep(60)
        cutoff = time.time() - _SESSION_TIMEOUT
        with _users_lock:
            stale = [k for k, v in _online_users.items() if v < cutoff]
            for k in stale:
                del _online_users[k]


threading.Thread(target=_cleanup_stale_sessions, daemon=True).start()


def _update_online_count(request: gr.Request):
    session_id = getattr(request, "session_hash", None) or str(id(request))
    with _users_lock:
        _online_users[session_id] = time.time()
        count = len(_online_users)
    return f"当前在线：{count} 人"


def _normalize_chatbot_history(hist):
    """Gradio 4.x + type='tuples'：每条须为 length==2 的 list（[user, bot]），不能是 tuple。"""
    if hist is None:
        return []
    if isinstance(hist, tuple):
        hist = list(hist)
    if not isinstance(hist, list):
        return []
    out = []
    for row in hist:
        if row is None:
            continue
        if isinstance(row, tuple):
            row = list(row)
        if isinstance(row, list) and len(row) >= 2:
            u, b = row[0], row[1]
            out.append(["" if u is None else str(u), "" if b is None else str(b)])
    return out


# --- Auth event handlers ---

def handle_login(username: str, password: str, request: gr.Request):
    ip = getattr(getattr(request, "client", None), "host", "")
    ok, reason = login_user(username, password, client_ip=ip)
    if not ok:
        return (
            "",
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=f"登录失败：{reason}"),
            gr.update(value=""),
            gr.update(value=""),
            [],
            0,
        )
    history = _normalize_chatbot_history(load_chat_history(username))
    turn_idx = get_next_turn_index(username)
    return (
        username,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=f"欢迎，{username}！"),
        history,
        turn_idx,
    )


def handle_register(username: str, password: str, request: gr.Request):
    """注册成功后自动登录，直接跳转主界面。"""
    ip = getattr(getattr(request, "client", None), "host", "")
    logger.info("[register] 用户 %s 请求注册，IP=%s", username, ip)
    ok, reason = register_user(username, password, bound_ip=ip)
    if not ok:
        logger.warning("[register] 注册失败: user=%s reason=%s", username, reason)
        # 错误提示写在 reg_msg，用户停留在「注册」Tab 时也能看到（勿只写 login_msg）
        return (
            "",
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=f"注册失败：{reason}"),
            gr.update(value=""),
            [],
            0,
        )
    logger.info("[register] 注册成功: user=%s，自动登录", username)
    history = _normalize_chatbot_history(load_chat_history(username))
    turn_idx = get_next_turn_index(username)
    return (
        username,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=f"欢迎，{username}！"),
        history,
        turn_idx,
    )


def handle_auto_login(saved_user: str):
    """页面加载时从 cookie/localStorage 恢复登录态（不校验密码，仅恢复 UI 状态）。"""
    import traceback as tb
    try:
        saved_user = (saved_user or "").strip()
        logger.info(f"[handle_auto_login] ========== 开始自动登录 ==========")
        logger.info(f"[handle_auto_login] 接收到的 saved_user: {repr(saved_user)}, type={type(saved_user)}, len={len(saved_user)}")
        
        if not saved_user:
            logger.info("[handle_auto_login] 无保存的用户，返回登录界面")
            return (
                "",
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                [],
                0,
            )
        
        logger.info(f"[handle_auto_login] 开始查询数据库中的用户: {saved_user}")
        from database.qa_log import SessionLocal as _SL, User as _U
        session = _SL()
        try:
            user = session.query(_U).filter_by(username=saved_user).first()
            if not user:
                logger.warning(f"[handle_auto_login] ❌ 用户不存在: {saved_user}")
                return (
                    "",
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=""),
                    [],
                    0,
                )
            logger.info(f"[handle_auto_login] ✓ 数据库中找到用户: {saved_user}")
        finally:
            session.close()
        
        logger.info(f"[handle_auto_login] 开始加载聊天历史...")
        history = _normalize_chatbot_history(load_chat_history(saved_user))
        turn_idx = get_next_turn_index(saved_user)
        logger.info(f"[handle_auto_login] ✓✓✓ 自动登录成功 ✓✓✓")
        logger.info(f"[handle_auto_login] user={saved_user}, history_len={len(history)}, turn_idx={turn_idx}")
        logger.info(f"[handle_auto_login] 返回: 隐藏登录界面, 显示问答界面")
        return (
            saved_user,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=f"欢迎，{saved_user}！"),
            history,
            turn_idx,
        )
    except Exception as ex:
        logger.error(f"[handle_auto_login] ❌ 异常: {ex}")
        logger.error(tb.format_exc())
        return (
            "",
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            [],
            0,
        )


def _internal_api_base() -> str:
    port = (
        os.environ.get("API_PORT")
        or os.environ.get("BACKEND_PORT")
        or "8001"
    ).strip() or "8001"
    return f"http://127.0.0.1:{port}"


def handle_logout():
    return (
        "",
        gr.update(visible=True),
        gr.update(visible=False),
        [],
        0,
    )


def handle_ip_change_request(real_name: str, reason: str, current_user_val: str, request: gr.Request):
    if not current_user_val:
        return gr.update(value="请先登录")
    real_name = real_name.strip()
    if not real_name:
        return gr.update(value="请填写真实姓名")
    new_ip = getattr(getattr(request, "client", None), "host", "")
    if not new_ip:
        return gr.update(value="无法获取当前 IP，请稍后重试")
    ok, msg = submit_ip_change(current_user_val, real_name, new_ip, reason)
    if ok:
        return gr.update(value=f"申请已提交，当前 IP：{new_ip}，请等待管理员审批。")
    return gr.update(value=f"提交失败：{msg}")


# --- Streaming answer helpers ---

def _loading_generator(question, chat_history, mode, answer_fn, username, turn_idx, *args):
    """Streams a loading indicator while the answer is computed, then yields the result."""
    logger.info(f"[_loading_generator] 开始处理问题: mode={mode}, user={username}, question={question[:50]}...")
    chat_history = _normalize_chatbot_history(chat_history or [])
    
    result_container = [None]
    error_container = [None]
    done_event = threading.Event()

    def _real_run():
        try:
            logger.info(f"[_loading_generator] 后台线程开始执行 answer_fn...")
            result_container[0] = answer_fn(question, chat_history, *args)
            logger.info(f"[_loading_generator] 后台线程完成，result type: {type(result_container[0])}")
        except Exception as e:
            logger.error(f"[_loading_generator] 后台线程异常: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            error_container[0] = e
        finally:
            done_event.set()
            logger.info(f"[_loading_generator] 后台线程已设置 done_event")

    thread = threading.Thread(target=_real_run, daemon=True)
    thread.start()
    logger.info(f"[_loading_generator] 后台线程已启动，开始等待...")

    start = time.time()
    try:
        max_wait = float(os.environ.get("GRADIO_QA_MAX_WAIT_SEC", "900"))
    except ValueError:
        max_wait = 900.0

    step_msgs = [
        "🔍 **第一步：检索知识库**\n正在搜索相关文档...",
        "📚 **第二步：分析文档**\n已找到相关内容，正在提取关键信息...",
        "💭 **第三步：理解问题**\n正在结合上下文理解您的问题...",
        "🧠 **第四步：组织答案**\n正在整理答案逻辑和结构...",
        "✍️ **第五步：生成回答**\n正在生成详细完整的回答..."
    ]
    
    while not done_event.wait(timeout=0.5):
        elapsed = int(time.time() - start)
        if elapsed >= max_wait:
            logger.error("[_loading_generator] 超过 GRADIO_QA_MAX_WAIT_SEC=%s 秒仍未完成", max_wait)
            yield (
                "",
                _normalize_chatbot_history(
                    chat_history
                    + [[question, f"⚠️ 已超过最长等待时间（{int(max_wait)} 秒），请稍后重试。"]]
                ),
                turn_idx,
            )
            return
        
        # 根据时间显示不同的思考阶段（每3秒切换一个阶段）
        step_idx = min(elapsed // 3, len(step_msgs) - 1)
        loading_msg = f"{step_msgs[step_idx]}\n\n*思考中... 已用时 {elapsed} 秒*"
        yield "", _normalize_chatbot_history(chat_history + [[question, loading_msg]]), turn_idx

    logger.info(f"[_loading_generator] done_event 已触发，开始处理结果...")
    
    if error_container[0]:
        logger.error(f"[_loading_generator] 检测到错误: {type(error_container[0]).__name__}: {error_container[0]}")
        yield "", _normalize_chatbot_history(
            chat_history + [[question, "抱歉，系统处理您的问题时出现了错误，请稍后重试。"]]
        ), turn_idx
        return

    logger.info(f"[_loading_generator] result_container[0] = {result_container[0]}")
    if result_container[0] is None:
        logger.error(f"[_loading_generator] result_container[0] 为 None！")
        yield "", _normalize_chatbot_history(
            chat_history + [[question, "抱歉，系统未能生成回答。"]]
        ), turn_idx
        return
    
    _, result = result_container[0]
    logger.info(f"[_loading_generator] 解包结果: result type={type(result)}, len={len(result) if isinstance(result, (list, tuple)) else 'N/A'}")
    
    if result and len(result) > 0:
        last_answer = result[-1][1] if isinstance(result, list) else str(result)
        try:
            save_qa(question, last_answer, mode)
        except Exception:
            pass
        if username:
            try:
                save_chat_turn(username, question, last_answer, mode, turn_idx)
            except Exception:
                pass
    
    final_history = _normalize_chatbot_history(result if isinstance(result, (list, tuple)) else [])
    logger.info(f"[_loading_generator] 最终返回历史记录，条数: {len(final_history)}")
    yield "", final_history, turn_idx + 1


def chat_qa_chain_self_answer_streaming(question, chat_history, model, embedding, temperature,
                                        top_k, history_len, current_user, turn_counter,
                                        file_path=DEFAULT_DB_PATH, persist_path=DEFAULT_PERSIST_PATH):
    if not question or len(question) < 1:
        yield "", _normalize_chatbot_history(chat_history), turn_counter
        return
    yield from _loading_generator(
        question, chat_history, "知识库问答（带记忆）",
        lambda q, h, *a: model_center.chat_qa_chain_self_answer(
            q, h, model, embedding, temperature, top_k, history_len, file_path, persist_path),
        current_user, turn_counter,
    )


def qa_chain_self_answer_streaming(question, chat_history, model, embedding, temperature,
                                   top_k, current_user, turn_counter,
                                   file_path=DEFAULT_DB_PATH, persist_path=DEFAULT_PERSIST_PATH):
    if not question or len(question) < 1:
        yield "", _normalize_chatbot_history(chat_history), turn_counter
        return
    yield from _loading_generator(
        question, chat_history, "知识库问答",
        lambda q, h, *a: model_center.qa_chain_self_answer(
            q, h, model, embedding, temperature, top_k, file_path, persist_path),
        current_user, turn_counter,
    )


def respond_streaming(message, chat_history, llm, history_len, temperature,
                      current_user, turn_counter, max_tokens=2048):
    if not message or len(message) < 1:
        yield "", _normalize_chatbot_history(chat_history), turn_counter
        return
    yield from _loading_generator(
        message, chat_history, "直接对话",
        lambda q, h, *a: respond(q, h, llm, history_len, temperature, max_tokens),
        current_user, turn_counter,
    )


# --- CSS ---
custom_css = """
.gradio-container { max-width: 900px !important; margin: auto; }
#title-text { text-align: center; margin-bottom: 4px; }
#subtitle-text { text-align: center; color: #666; margin-top: 0; }
#online-count { text-align: right; color: #888; font-size: 0.85em; margin: 0 0 8px 0; }
#auth-title { text-align: center; }
#login-msg { color: #c00; font-size: 0.9em; min-height: 1.4em; }
#reg-msg { color: #c00; font-size: 0.9em; min-height: 1.4em; }
#welcome-msg { color: #555; font-size: 0.9em; margin-bottom: 4px; }
#ip-change-msg { color: #555; font-size: 0.9em; min-height: 1.4em; }
footer { display: none !important; }
.built-with { display: none !important; }
"""

demo = gr.Blocks(title="北京交通飞拳", css=custom_css)

with demo:
    current_user = gr.State("")
    turn_counter = gr.State(0)
    saved_user_state = gr.State("")  # 用于JS传递自动登录的用户名

    # ---- Auth panel ----
    with gr.Column(visible=True) as auth_panel:
        gr.Markdown("## 北京交通飞拳", elem_id="auth-title")
        gr.Markdown("智能知识库问答助手", elem_id="subtitle-text")
        with gr.Tabs():
            with gr.Tab("登录"):
                login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                login_password = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
                login_btn = gr.Button("登录", variant="primary")
                login_msg = gr.Markdown("", elem_id="login-msg")
            with gr.Tab("注册"):
                gr.Markdown("*注册后即可免费使用问答功能。*")
                reg_username = gr.Textbox(label="用户名", placeholder="请输入用户名（至少 2 位）")
                reg_password = gr.Textbox(label="密码", placeholder="请输入密码（至少 6 位）", type="password")
                reg_btn = gr.Button("注册", variant="secondary")
                reg_msg = gr.Markdown("", elem_id="reg-msg")

    # ---- Chat panel ----
    with gr.Column(visible=False) as chat_panel:
        gr.Markdown("# 北京交通飞拳", elem_id="title-text")
        gr.Markdown("智能知识库问答助手", elem_id="subtitle-text")
        online_count_display = gr.Markdown("当前在线：- 人", elem_id="online-count")
        welcome_msg = gr.Markdown("", elem_id="welcome-msg")

        llm = gr.State(INIT_LLM)
        embeddings = gr.State(INIT_EMBEDDING_MODEL)
        temperature = gr.State(0.01)
        top_k = gr.State(3)
        history_len = gr.State(3)

        chatbot = gr.Chatbot(height=480, type="tuples")
        msg = gr.Textbox(label="请输入您的问题", placeholder="在这里输入问题，按 Enter 发送...")

        with gr.Row():
            db_with_his_btn = gr.Button("知识库问答（带记忆）", variant="primary")
            db_wo_his_btn = gr.Button("知识库问答")
            llm_btn = gr.Button("直接对话")
            clear = gr.ClearButton(components=[chatbot], value="清空对话")

        with gr.Accordion("申请更换绑定 IP", open=False):
            gr.Markdown("提交后管理员审批通过，新 IP 即可登录。当前 IP 由系统自动获取，无需手填。")
            ip_change_name = gr.Textbox(label="真实姓名", placeholder="请填写您的真实姓名")
            ip_change_reason = gr.Textbox(label="变更原因（选填）", placeholder="例如：更换宽带、公司网络变更")
            ip_change_btn = gr.Button("提交申请", variant="secondary")
            ip_change_msg = gr.Markdown("", elem_id="ip-change-msg")

        logout_btn = gr.Button("退出登录", variant="stop")

    # ---- Event wiring ----
    _auth_outputs = [
        current_user,
        auth_panel,
        chat_panel,
        login_msg,
        reg_msg,
        welcome_msg,
        chatbot,
        turn_counter,
    ]

    ip_change_btn.click(handle_ip_change_request,
                        inputs=[ip_change_name, ip_change_reason, current_user],
                        outputs=[ip_change_msg])

    _chat_inputs_base = [msg, chatbot, llm, embeddings, temperature, top_k, history_len, current_user, turn_counter]
    _chat_outputs = [msg, chatbot, turn_counter]

    db_with_his_btn.click(chat_qa_chain_self_answer_streaming,
                          inputs=_chat_inputs_base, outputs=_chat_outputs)
    db_wo_his_btn.click(qa_chain_self_answer_streaming,
                        inputs=[msg, chatbot, llm, embeddings, temperature, top_k, current_user, turn_counter],
                        outputs=_chat_outputs)
    llm_btn.click(respond_streaming,
                  inputs=[msg, chatbot, llm, history_len, temperature, current_user, turn_counter],
                  outputs=_chat_outputs)
    msg.submit(chat_qa_chain_self_answer_streaming,
               inputs=_chat_inputs_base, outputs=_chat_outputs)

    # Online count: fires immediately on load, then every 30 s
    demo.load(_update_online_count, inputs=None, outputs=online_count_display)
    demo.load(_update_online_count, inputs=None, outputs=online_count_display, every=30)

    # --- 页面加载时恢复登录状态和填充账号密码 ---
    demo.load(
        lambda: (logger.info("[demo.load] lambda 被调用，返回空字符串"), "")[1],  # 先返回空字符串，JS会填充实际值
        inputs=None, 
        outputs=[saved_user_state],
        js="""() => {
            console.log('[自动登录JS] ========== 页面加载，开始读取用户状态 ==========');
            // 读取保存的用户名（优先 localStorage，其次 cookie）
            try {
                var saved = localStorage.getItem('bjtu_user');
                console.log('[自动登录JS] localStorage.bjtu_user =', saved);
                if (saved) {
                    console.log('[自动登录JS] ✓ 从 localStorage 读取到用户:', saved);
                    return saved;
                }
            } catch(e) {
                console.error('[自动登录JS] ❌ localStorage 读取失败:', e);
            }
            var cookies = document.cookie;
            console.log('[自动登录JS] document.cookie =', cookies);
            var m = document.cookie.match(/(?:^|;\\s*)bjtu_user=([^;]*)/);
            console.log('[自动登录JS] cookie match result =', m);
            if (m && m[1]) {
                var decoded = decodeURIComponent(m[1]);
                console.log('[自动登录JS] ✓ 从 cookie 读取到用户:', decoded);
                return decoded;
            }
            console.log('[自动登录JS] ❌ 未找到保存的用户，返回空字符串');
            return '';
        }"""
    ).then(
        handle_auto_login,
        inputs=[saved_user_state],
        outputs=_auth_outputs
    )
    
    # 自动填充用户名和密码（记住密码功能）
    demo.load(lambda: [None, None], inputs=None, outputs=[login_username, login_password],
              js="""() => {
                  try {
                      var savedUser = localStorage.getItem('bjtu_saved_username') || '';
                      var savedPass = localStorage.getItem('bjtu_saved_password') || '';
                      return [savedUser, savedPass];
                  } catch(e) {
                      return ['', ''];
                  }
              }""")

    # --- 登录：成功后保存账号密码（记住密码）和用户状态 ---
    _login_ev = login_btn.click(handle_login, inputs=[login_username, login_password], outputs=_auth_outputs)
    login_username.submit(handle_login, inputs=[login_username, login_password], outputs=_auth_outputs)
    _login_ev.then(
        lambda user, username, password: None,
        inputs=[current_user, login_username, login_password], outputs=[],
        js="""(user, username, password) => {
            if (user) {
                // 保存用户状态（30天过期）
                document.cookie = 'bjtu_user=' + encodeURIComponent(user) + '; path=/; max-age=2592000';
                try {
                    localStorage.setItem('bjtu_user', user);
                    // 记住密码功能：保存用户名和密码
                    localStorage.setItem('bjtu_saved_username', username);
                    localStorage.setItem('bjtu_saved_password', password);
                } catch(e) {}
            }
        }"""
    )

    # --- 注册：成功后保存账号密码和用户状态 ---
    _reg_ev = reg_btn.click(handle_register,
                   inputs=[reg_username, reg_password],
                   outputs=_auth_outputs)
    _reg_ev.then(
        lambda user, username, password: None,
        inputs=[current_user, reg_username, reg_password], outputs=[],
        js="""(user, username, password) => {
            if (user) {
                document.cookie = 'bjtu_user=' + encodeURIComponent(user) + '; path=/; max-age=2592000';
                try {
                    localStorage.setItem('bjtu_user', user);
                    // 记住密码
                    localStorage.setItem('bjtu_saved_username', username);
                    localStorage.setItem('bjtu_saved_password', password);
                } catch(e) {}
            }
        }"""
    )

    # --- 退出登录：清除所有保存的数据 ---
    _logout_ev = logout_btn.click(handle_logout, inputs=[], outputs=[current_user, auth_panel, chat_panel, chatbot, turn_counter])
    _logout_ev.then(
        lambda *a: None, inputs=[], outputs=[],
        js="""() => {
            document.cookie = 'bjtu_user=; path=/; max-age=0';
            try {
                localStorage.removeItem('bjtu_user');
                // 退出时不删除保存的密码，下次登录可继续使用
                // 如需清除密码，取消注释下面两行：
                // localStorage.removeItem('bjtu_saved_username');
                // localStorage.removeItem('bjtu_saved_password');
            } catch(e) {}
        }"""
    )
    clear.click(model_center.clear_history)

print("Gradio 服务启动中（使用官方 launch 模式）...")

_server_port_raw = (
    os.environ.get("FRONTEND_PORT")
    or os.environ.get("GRADIO_SERVER_PORT")
    or "7860"
).strip() or "7860"
server_port = int(_server_port_raw)
server_host = os.environ.get("GRADIO_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
logger.info(
    "Gradio 将监听 %s:%s；内部 API 基址 %s",
    server_host,
    server_port,
    _internal_api_base(),
)

print(f"✅ Gradio 前端服务启动！")
print(f"   监听: {server_host}:{server_port}")
if server_host in ("127.0.0.1", "localhost", "::1"):
    print(f"   本机访问: http://127.0.0.1:{server_port}  或  http://localhost:{server_port}")
else:
    print(f"   请在浏览器使用本机公网 IP 或域名访问该端口（生产环境务必走 HTTPS 反向代理）。")
print(f"\n提示：按 Ctrl+C 停止服务\n")
print("=" * 60)
logger.info("所有组件初始化完成，启动 Gradio 服务...")

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

demo.queue().launch(
    server_name=server_host,
    server_port=server_port,
    share=False,
    inbrowser=False,
)
