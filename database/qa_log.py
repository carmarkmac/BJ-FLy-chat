import os
import hashlib
import secrets
import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_PROJECT_ROOT, "qa_log.db")

engine = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class QARecord(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    mode = Column(String(32), nullable=False)
    is_unanswered = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    salt          = Column(String(32), nullable=False)
    bound_ip      = Column(String(64), nullable=True)   # NULL = legacy account, skip IP check
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)


class InviteCode(Base):
    __tablename__ = "invite_codes"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    code       = Column(String(64), unique=True, nullable=False, index=True)
    used_by    = Column(String(64), nullable=True)
    used_at    = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    username   = Column(String(64), nullable=False, index=True)
    role       = Column(String(8), nullable=False)   # "user" or "bot"
    content    = Column(Text, nullable=False)
    turn_index = Column(Integer, nullable=False)
    mode       = Column(String(32), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class IpChangeRequest(Base):
    __tablename__ = "ip_change_requests"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    username   = Column(String(64), nullable=False, index=True)
    real_name  = Column(String(64), nullable=False)
    new_ip     = Column(String(64), nullable=False)
    reason     = Column(Text, nullable=True)
    status     = Column(String(16), nullable=False, default="pending")  # pending/approved/rejected
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    handled_at = Column(DateTime, nullable=True)


Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# QA logging (existing)
# ---------------------------------------------------------------------------

def save_qa(question: str, answer: str, mode: str):
    """Persist a Q&A pair. Marks is_unanswered when the KB has no relevant info."""
    if not question or len(question.strip()) < 5:
        return
    _NO_INFO_SIGNALS = ["暂无该问题的相关信息", "知识库中暂无", "没有相关信息", "无法回答", "抱歉，知识库"]
    is_unanswered = any(s in answer for s in _NO_INFO_SIGNALS)
    session = SessionLocal()
    try:
        session.add(QARecord(
            question=question.strip(),
            answer=answer,
            mode=mode,
            is_unanswered=is_unanswered,
        ))
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# User auth
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def register_user(username: str, password: str, bound_ip: str = ""):
    """
    Returns (True, "") on success.
    Returns (False, reason) on failure.
    每个 IP 只能注册一次。
    """
    username = username.strip()
    if not username or len(username) < 2:
        logger.info("[register] 校验失败: 用户名太短 '%s'", username)
        return False, "用户名至少 2 个字符"
    if len(username) > 64:
        return False, "用户名过长"
    if not password or len(password) < 6:
        logger.info("[register] 校验失败: 密码太短")
        return False, "密码至少 6 位"

    session = SessionLocal()
    try:
        if session.query(User).filter_by(username=username).first():
            logger.info("[register] 校验失败: 用户名 '%s' 已存在", username)
            return False, "用户名已存在"
        if bound_ip:
            existing = session.query(User).filter_by(bound_ip=bound_ip).first()
            if existing:
                logger.info("[register] 校验失败: IP '%s' 已绑定用户 '%s'", bound_ip, existing.username)
                return False, "该 IP 已注册过账号，每个网络环境仅允许注册一次"

        salt = secrets.token_hex(16)
        pw_hash = _hash_password(password, salt)
        session.add(User(username=username, password_hash=pw_hash, salt=salt,
                         bound_ip=bound_ip or None))
        session.commit()
        logger.info("[register] 注册成功: user=%s ip=%s", username, bound_ip)
        return True, ""
    except Exception as e:
        session.rollback()
        logger.error("[register] 数据库异常: %s", e)
        return False, f"注册失败：{e}"
    finally:
        session.close()


def login_user(username: str, password: str, client_ip: str = ""):
    """
    Returns (True, "") on success.
    Returns (False, reason) on failure.
    If user has no bound_ip yet, auto-binds to current client_ip on first login.
    """
    username = username.strip()
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(username=username).first()
        if not user:
            return False, "用户名或密码错误"
        if _hash_password(password, user.salt) != user.password_hash:
            return False, "用户名或密码错误"
        if user.bound_ip:
            # 已绑定 IP：必须一致
            if client_ip and user.bound_ip != client_ip:
                return False, "当前 IP 与绑定 IP 不符，无法登录。如需更换绑定 IP，请提交变更申请。"
        else:
            # 首次登录，自动绑定当前 IP
            if client_ip:
                user.bound_ip = client_ip
                session.commit()
        return True, ""
    except Exception:
        session.rollback()
        return True, ""   # 登录本身成功，绑定失败不阻断
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Chat history persistence
# ---------------------------------------------------------------------------

def get_next_turn_index(username: str) -> int:
    session = SessionLocal()
    try:
        from sqlalchemy import func
        result = session.query(func.max(ChatHistory.turn_index)).filter_by(username=username).scalar()
        return 0 if result is None else result + 1
    finally:
        session.close()


def save_chat_turn(username: str, question: str, answer: str, mode: str, turn_index: int):
    if not username:
        return
    session = SessionLocal()
    try:
        session.add(ChatHistory(username=username, role="user",
                                content=question, turn_index=turn_index, mode=mode))
        session.add(ChatHistory(username=username, role="bot",
                                content=answer, turn_index=turn_index, mode=mode))
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


def load_chat_history(username: str):
    """Returns [(question, answer), ...] ordered by turn_index."""
    session = SessionLocal()
    try:
        rows = (session.query(ChatHistory)
                .filter_by(username=username)
                .order_by(ChatHistory.turn_index, ChatHistory.id)
                .all())
        turns: dict = {}
        for row in rows:
            idx = row.turn_index
            if idx not in turns:
                turns[idx] = {"user": "", "bot": ""}
            turns[idx][row.role] = row.content
        return [(turns[i]["user"], turns[i]["bot"]) for i in sorted(turns)]
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Invite code management (admin utility)
# ---------------------------------------------------------------------------

def generate_invite_code() -> str:
    """Creates a new unused invite code, persists it, and returns the code string."""
    code = secrets.token_urlsafe(12)
    session = SessionLocal()
    try:
        session.add(InviteCode(code=code))
        session.commit()
        return code
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# IP change request management
# ---------------------------------------------------------------------------

def submit_ip_change(username: str, real_name: str, new_ip: str, reason: str = ""):
    """
    User submits a request to rebind their account to a new IP.
    Returns (True, "") on success, (False, reason) on failure.
    """
    real_name = real_name.strip()
    if not real_name:
        return False, "真实姓名不能为空"
    if not new_ip:
        return False, "无法获取当前 IP"
    session = SessionLocal()
    try:
        # Prevent duplicate pending requests from the same user
        existing = (session.query(IpChangeRequest)
                    .filter_by(username=username, status="pending")
                    .first())
        if existing:
            return False, "您已有一条待审批的申请，请等待管理员处理后再提交"
        session.add(IpChangeRequest(
            username=username,
            real_name=real_name,
            new_ip=new_ip,
            reason=reason.strip() or None,
        ))
        session.commit()
        return True, ""
    except Exception as e:
        session.rollback()
        return False, f"提交失败：{e}"
    finally:
        session.close()


def approve_ip_change(request_id: int):
    """
    Admin approves an IP change request.
    Updates User.bound_ip and marks the request as approved.
    Returns (True, "") on success, (False, reason) on failure.
    """
    session = SessionLocal()
    try:
        req = session.query(IpChangeRequest).filter_by(id=request_id).first()
        if not req:
            return False, f"申请 ID {request_id} 不存在"
        if req.status != "pending":
            return False, f"申请状态为 {req.status}，无法重复审批"
        user = session.query(User).filter_by(username=req.username).first()
        if not user:
            return False, f"用户 {req.username} 不存在"
        user.bound_ip = req.new_ip
        req.status = "approved"
        req.handled_at = datetime.utcnow()
        session.commit()
        return True, ""
    except Exception as e:
        session.rollback()
        return False, f"审批失败：{e}"
    finally:
        session.close()


def reject_ip_change(request_id: int):
    """Admin rejects an IP change request."""
    session = SessionLocal()
    try:
        req = session.query(IpChangeRequest).filter_by(id=request_id).first()
        if not req:
            return False, f"申请 ID {request_id} 不存在"
        if req.status != "pending":
            return False, f"申请状态为 {req.status}，无法操作"
        req.status = "rejected"
        req.handled_at = datetime.utcnow()
        session.commit()
        return True, ""
    except Exception as e:
        session.rollback()
        return False, f"操作失败：{e}"
    finally:
        session.close()


def list_pending_ip_changes():
    """Returns all pending IP change requests as a list of dicts."""
    session = SessionLocal()
    try:
        rows = (session.query(IpChangeRequest)
                .filter_by(status="pending")
                .order_by(IpChangeRequest.created_at)
                .all())
        return [
            {"id": r.id, "username": r.username, "real_name": r.real_name,
             "new_ip": r.new_ip, "reason": r.reason, "created_at": str(r.created_at)}
            for r in rows
        ]
    finally:
        session.close()
