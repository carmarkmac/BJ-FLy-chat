from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, validator
from starlette.middleware.trustedhost import TrustedHostMiddleware
from typing import Optional
import hmac
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_chain.QA_chain_self import QA_chain_self
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_trusted_hosts = os.environ.get("TRUSTED_HOSTS", "").strip()
if _trusted_hosts:
    hosts = [h.strip() for h in _trusted_hosts.split(",") if h.strip()]
    if hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
        logger.info("TrustedHostMiddleware 已启用: %s", hosts)

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_db", "chroma")
_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_db")
_API_TOKEN = os.environ.get("API_TOKEN", "")
if _API_TOKEN and len(_API_TOKEN) < 16:
    logger.warning("API_TOKEN 长度建议至少 16 字符，当前过短不利于抵抗在线猜测。")


def _api_token_matches(provided: Optional[str]) -> bool:
    if provided is None:
        provided = ""
    try:
        return hmac.compare_digest(
            provided.encode("utf-8"),
            _API_TOKEN.encode("utf-8"),
        )
    except Exception:
        return False


class Item(BaseModel):
    prompt: str
    model: str = "deepseek-chat"
    temperature: float = 0.1
    embedding: str = "m3e"
    top_k: int = 5

    @validator("prompt")
    def prompt_not_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("prompt cannot be empty")
        if len(v) > 2000:
            raise ValueError("prompt too long (max 2000 chars)")
        return v.strip()

    @validator("temperature")
    def temperature_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("temperature must be between 0 and 1")
        return v

    @validator("top_k")
    def top_k_range(cls, v):
        if not 1 <= v <= 10:
            raise ValueError("top_k must be between 1 and 10")
        return v


@app.post("/")
@limiter.limit("20/minute")
async def get_response(request: Request, item: Item, x_api_token: str = Header(None)):
    if not _API_TOKEN:
        raise HTTPException(status_code=500, detail="Server API token not configured")
    if not _api_token_matches(x_api_token):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        chain = QA_chain_self(
            model=item.model,
            temperature=item.temperature,
            top_k=item.top_k,
            file_path=_FILE_PATH,
            persist_path=_DB_PATH,
            embedding=item.embedding,
        )
        return {"answer": chain.answer(question=item.prompt)}
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
