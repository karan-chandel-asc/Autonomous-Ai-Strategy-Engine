import os
import re
import time
import threading
from langchain_groq import ChatGroq

# Groq retired meta-llama/llama-4-scout-17b-16e-instruct on 2026-07-17.
# Note: llama-3.3-70b-versatile is scheduled for shutdown on 2026-08-16.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

_CHAT_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "400"))
_JSON_MAX_TOKENS = int(os.environ.get("GROQ_JSON_MAX_TOKENS", "250"))
_RATE_LIMIT_ATTEMPTS = int(os.environ.get("GROQ_RATE_LIMIT_ATTEMPTS", "8"))
_SERIALIZE_LLM = os.environ.get("GROQ_SERIALIZE_LLM", "1") == "1"
_LLM_LOCK_KEY = os.environ.get("GROQ_LLM_LOCK_KEY", "groq:llm_lock")
_LLM_LOCK_TTL = int(os.environ.get("GROQ_LLM_LOCK_TTL", "120"))
_KEY_INDEX_REDIS = os.environ.get("GROQ_KEY_INDEX_REDIS", "groq:api_key_index")

_local_key_index = 0
_local_key_lock = threading.Lock()


def get_groq_model_name() -> str:
    return os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def get_groq_api_keys() -> list[str]:
    """Load Groq keys from GROQ_API_KEYS=k1,k2 or GROQ_API_KEY / GROQ_API_KEY_2 / …"""
    keys: list[str] = []
    multi = os.environ.get("GROQ_API_KEYS", "") or ""
    for part in multi.split(","):
        k = part.strip().strip('"').strip("'")
        if k and k not in keys:
            keys.append(k)
    for name in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4"):
        raw = os.environ.get(name, "") or ""
        k = raw.strip().strip('"').strip("'")
        if k and k not in keys:
            keys.append(k)
    return keys


def _redis_client():
    try:
        import redis as redis_lib
        return redis_lib.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    except Exception:
        return None


def _get_key_index(n: int) -> int:
    if n <= 0:
        return 0
    r = _redis_client()
    if r is not None:
        try:
            return int(r.get(_KEY_INDEX_REDIS) or 0) % n
        except Exception:
            pass
    with _local_key_lock:
        return _local_key_index % n


def _set_key_index(idx: int, n: int) -> None:
    global _local_key_index
    if n <= 0:
        return
    idx = idx % n
    r = _redis_client()
    if r is not None:
        try:
            r.set(_KEY_INDEX_REDIS, idx)
        except Exception:
            pass
    with _local_key_lock:
        _local_key_index = idx


def get_current_groq_api_key() -> str | None:
    keys = get_groq_api_keys()
    if not keys:
        return None
    return keys[_get_key_index(len(keys))]


def rotate_groq_api_key() -> str | None:
    """Move to the next Groq API key. Returns the new key (or None if none configured)."""
    keys = get_groq_api_keys()
    if not keys:
        return None
    nxt = (_get_key_index(len(keys)) + 1) % len(keys)
    _set_key_index(nxt, len(keys))
    return keys[nxt]


def _parse_retry_wait_seconds(msg: str, attempt: int, *, switched_key: bool) -> float:
    """Parse Groq retry hint. If we switched keys, wait briefly; else wait for TPM refill."""
    if switched_key:
        return 0.5
    match_ms = re.search(r"try again in ([\d.]+)\s*ms", msg, re.I)
    match_s = re.search(r"try again in ([\d.]+)\s*s", msg, re.I)
    if match_ms:
        suggested = float(match_ms.group(1)) / 1000.0
    elif match_s:
        suggested = float(match_s.group(1))
    else:
        suggested = float(2 ** attempt + 2)
    return max(suggested + 0.5, 10.0, min(2 ** attempt + 5, 45))


def _acquire_llm_lock(timeout: float = 90.0):
    if not _SERIALIZE_LLM:
        return None
    try:
        r = _redis_client()
        if r is None:
            return None
        token = f"{os.getpid()}-{time.time()}"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if r.set(_LLM_LOCK_KEY, token, nx=True, ex=_LLM_LOCK_TTL):
                return (r, token)
            time.sleep(0.4)
    except Exception:
        return None
    return None


def _release_llm_lock(handle) -> None:
    if not handle:
        return
    r, token = handle
    try:
        if r.get(_LLM_LOCK_KEY) == token:
            r.delete(_LLM_LOCK_KEY)
    except Exception:
        pass


def _clone_llm_with_current_key(llm) -> ChatGroq:
    """Rebuild ChatGroq with the currently selected API key."""
    model = getattr(llm, "model_name", None) or get_groq_model_name()
    kwargs = {
        "model": model,
        "temperature": getattr(llm, "temperature", 0.6),
        "max_tokens": getattr(llm, "max_tokens", _CHAT_MAX_TOKENS),
    }
    key = get_current_groq_api_key()
    if key:
        kwargs["api_key"] = key
    model_kwargs = getattr(llm, "model_kwargs", None) or {}
    if model_kwargs:
        kwargs["model_kwargs"] = dict(model_kwargs)
    return ChatGroq(**kwargs)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "rate_limit" in msg.lower()
        or "429" in msg
        or "RateLimitError" in type(exc).__name__
    )


def invoke_with_rate_limit_retry(llm, messages, attempts: int = _RATE_LIMIT_ATTEMPTS):
    """Invoke Groq; on 429 rotate to next API key, else wait for TPM refill."""
    last_exc = None
    keys = get_groq_api_keys()
    keys_tried_this_round = 0
    llm = _clone_llm_with_current_key(llm)
    lock = _acquire_llm_lock()
    try:
        for attempt in range(attempts):
            try:
                return llm.invoke(messages)
            except Exception as exc:
                last_exc = exc
                if not _is_rate_limit_error(exc) or attempt == attempts - 1:
                    raise

                switched = False
                if len(keys) > 1:
                    rotate_groq_api_key()
                    llm = _clone_llm_with_current_key(llm)
                    keys_tried_this_round += 1
                    switched = keys_tried_this_round < len(keys)
                    if keys_tried_this_round >= len(keys):
                        # All keys hit 429 this round — wait, then cycle again
                        keys_tried_this_round = 0

                wait = _parse_retry_wait_seconds(
                    str(exc), attempt, switched_key=switched
                )
                time.sleep(wait)
        raise last_exc
    finally:
        _release_llm_lock(lock)


def _build_chat_groq(**kwargs) -> ChatGroq:
    key = get_current_groq_api_key()
    if key:
        kwargs["api_key"] = key
    return ChatGroq(**kwargs)


class Lanchain_models():
    def __init__(self):
        pass

    def get_chat_model(self):
        return _build_chat_groq(
            model=get_groq_model_name(),
            temperature=0.6,
            max_tokens=_CHAT_MAX_TOKENS,
        )

    def get_json_chat_model(self):
        """Aggregation model — JSON mode, capped for free-tier TPM."""
        return _build_chat_groq(
            model=get_groq_model_name(),
            temperature=0.4,
            max_tokens=_JSON_MAX_TOKENS,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
