import logging
import os
import re
import time
import threading
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# Default if an agent has no mapping
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

# Free-tier-oriented map: spread agents across separate Groq TPM buckets.
# Limits (approx free plan): 8b=6k TPM/500k TPD, 20b=8k/200k, 120b=8k/200k,
# 70b=12k/100k, qwen3.6-27b=8k/200k.
AGENT_MODEL_MAP = {
    "executive_summary": "openai/gpt-oss-20b",        # concise narrative
    "market_analysis": "openai/gpt-oss-120b",         # heavier market reasoning
    "competitive_landscape": "qwen/qwen3.6-27b",      # structured competitor JSON
    "monetization_strategy": "llama-3.3-70b-versatile",  # numeric / unit economics
    "risk_assessment": "llama-3.1-8b-instant",        # high TPD, lighter schema
    "roadmap": "openai/gpt-oss-20b",                  # phased plan JSON
    "weakness_review": "llama-3.1-8b-instant",        # audit bullets
    # gpt-oss JSON mode often returns empty failed_generation — use Llama for agg
    "aggregation": "llama-3.1-8b-instant",
    "validate_query": "llama-3.1-8b-instant",         # tiny gatekeeper call
}

# Models that frequently fail Groq response_format=json_object
_JSON_MODE_UNRELIABLE = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
}

# Output caps matched to task size (keeps each model under its TPM).
AGENT_MAX_TOKENS = {
    "executive_summary": 350,
    "market_analysis": 400,
    "competitive_landscape": 400,
    "monetization_strategy": 350,
    "risk_assessment": 350,
    "roadmap": 400,
    "weakness_review": 350,
    "aggregation": 400,
    "validate_query": 80,
}

# On 429 for a model, try these next (different TPM buckets).
MODEL_FALLBACKS = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
]

_CHAT_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "400"))
_JSON_MAX_TOKENS = int(os.environ.get("GROQ_JSON_MAX_TOKENS", "250"))
_RATE_LIMIT_ATTEMPTS = int(os.environ.get("GROQ_RATE_LIMIT_ATTEMPTS", "8"))
# Off by default: agents already use different models (separate TPM). Locking made runs ~2x slower.
_SERIALIZE_LLM = os.environ.get("GROQ_SERIALIZE_LLM", "0") == "1"
_ROUND_ROBIN = os.environ.get("GROQ_ROUND_ROBIN", "1") == "1"
_LLM_LOCK_KEY = os.environ.get("GROQ_LLM_LOCK_KEY", "groq:llm_lock")
_LLM_LOCK_TTL = int(os.environ.get("GROQ_LLM_LOCK_TTL", "120"))
_KEY_INDEX_REDIS = os.environ.get("GROQ_KEY_INDEX_REDIS", "groq:api_key_index")

_local_key_index = 0
_local_key_lock = threading.Lock()


def get_groq_model_name(agent_name: str | None = None) -> str:
    """Resolve model for an agent; GROQ_MODEL env overrides everything if set."""
    override = os.environ.get("GROQ_MODEL", "").strip()
    if override:
        return override
    if agent_name and agent_name in AGENT_MODEL_MAP:
        return AGENT_MODEL_MAP[agent_name]
    return DEFAULT_GROQ_MODEL


def get_agent_max_tokens(agent_name: str | None = None, *, json_mode: bool = False) -> int:
    if agent_name and agent_name in AGENT_MAX_TOKENS:
        return AGENT_MAX_TOKENS[agent_name]
    return _JSON_MAX_TOKENS if json_mode else _CHAT_MAX_TOKENS


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


def _mask_key(key: str | None) -> str:
    if not key or len(key) < 12:
        return "none"
    return f"{key[:6]}...{key[-4:]}"


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


def rotate_groq_api_key(failed_key: str | None = None) -> str | None:
    """Move to the next Groq API key (skips the key that just failed when possible)."""
    keys = get_groq_api_keys()
    if not keys:
        return None
    if failed_key and failed_key in keys and len(keys) > 1:
        nxt = (keys.index(failed_key) + 1) % len(keys)
    else:
        nxt = (_get_key_index(len(keys)) + 1) % len(keys)
    _set_key_index(nxt, len(keys))
    logger.info(
        f"[Groq] rotated API key -> slot {nxt + 1}/{len(keys)} ({_mask_key(keys[nxt])})"
    )
    return keys[nxt]


def next_groq_api_key_for_call() -> str | None:
    """Pick key for this call. Round-robin advances index every call when enabled."""
    keys = get_groq_api_keys()
    if not keys:
        return None
    if not _ROUND_ROBIN or len(keys) == 1:
        return keys[_get_key_index(len(keys))]
    idx = _get_key_index(len(keys))
    key = keys[idx]
    _set_key_index(idx + 1, len(keys))
    return key


def _next_fallback_model(current: str) -> str | None:
    """Return next different model from fallback list."""
    ordered = []
    for m in MODEL_FALLBACKS:
        if m not in ordered:
            ordered.append(m)
    if current not in ordered:
        ordered.insert(0, current)
    try:
        i = ordered.index(current)
    except ValueError:
        return ordered[0] if ordered else None
    if len(ordered) <= 1:
        return None
    return ordered[(i + 1) % len(ordered)]


def _parse_retry_wait_seconds(msg: str, attempt: int, *, switched: bool) -> float:
    if switched:
        return 0.5
    match_combo = re.search(
        r"try again in (?:(\d+)\s*m)?\s*([\d.]+)\s*(ms|s)",
        msg,
        re.I,
    )
    if match_combo:
        minutes = float(match_combo.group(1) or 0)
        amount = float(match_combo.group(2))
        unit = match_combo.group(3).lower()
        suggested = minutes * 60.0 + (amount / 1000.0 if unit == "ms" else amount)
    else:
        suggested = float(2 ** attempt + 2)
    return max(min(suggested + 0.5, 20.0), 2.0)


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


def _build_llm(
    *,
    model: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    model_kwargs: dict | None = None,
) -> ChatGroq:
    kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if model_kwargs:
        kwargs["model_kwargs"] = dict(model_kwargs)
    return ChatGroq(**kwargs)


def _llm_params(llm) -> dict:
    return {
        "model": getattr(llm, "model_name", None) or get_groq_model_name(),
        "temperature": getattr(llm, "temperature", 0.6),
        "max_tokens": getattr(llm, "max_tokens", _CHAT_MAX_TOKENS),
        "model_kwargs": dict(getattr(llm, "model_kwargs", None) or {}),
    }


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "rate_limit" in msg.lower()
        or "429" in msg
        or "RateLimitError" in type(exc).__name__
    )


def _is_json_validate_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "json_validate_failed" in msg or "failed to validate json" in msg


def invoke_with_rate_limit_retry(llm, messages, attempts: int = _RATE_LIMIT_ATTEMPTS):
    """Invoke Groq; on 429 rotate key/model; on JSON-mode fail, drop JSON mode and retry."""
    last_exc = None
    keys = get_groq_api_keys()
    params = _llm_params(llm)
    call_key = next_groq_api_key_for_call()
    call_model = params["model"]
    model_kwargs = dict(params["model_kwargs"] or {})
    # gpt-oss family often returns empty content under json_object mode
    if call_model in _JSON_MODE_UNRELIABLE:
        model_kwargs.pop("response_format", None)

    llm = _build_llm(
        model=call_model,
        api_key=call_key,
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
        model_kwargs=model_kwargs or None,
    )
    logger.info(
        f"[Groq] call model={call_model} key={_mask_key(call_key)} "
        f"keys={len(keys)}"
    )

    keys_tried = 0
    models_tried = {call_model}
    lock = _acquire_llm_lock()
    try:
        for attempt in range(attempts):
            try:
                return llm.invoke(messages)
            except Exception as exc:
                last_exc = exc
                # JSON mode failure → retry without response_format (prompt already asks JSON)
                if _is_json_validate_error(exc):
                    if model_kwargs.get("response_format"):
                        model_kwargs.pop("response_format", None)
                        call_model = "llama-3.1-8b-instant"
                        models_tried.add(call_model)
                        llm = _build_llm(
                            model=call_model,
                            api_key=call_key,
                            temperature=params["temperature"],
                            max_tokens=max(params["max_tokens"], 400),
                            model_kwargs=None,
                        )
                        logger.warning(
                            "[Groq] json_validate_failed → retry without JSON mode "
                            f"model={call_model}"
                        )
                        continue
                    if attempt < attempts - 1:
                        nxt = _next_fallback_model(call_model) or "llama-3.1-8b-instant"
                        call_model = nxt
                        models_tried.add(call_model)
                        llm = _build_llm(
                            model=call_model,
                            api_key=call_key,
                            temperature=params["temperature"],
                            max_tokens=max(params["max_tokens"], 400),
                            model_kwargs=None,
                        )
                        continue
                    raise

                if not _is_rate_limit_error(exc) or attempt == attempts - 1:
                    raise

                switched = False
                # 1) Prefer another API key first
                if len(keys) > 1:
                    failed = call_key
                    new_key = rotate_groq_api_key(failed_key=failed)
                    if new_key and new_key != failed:
                        call_key = new_key
                        switched = True
                        keys_tried += 1

                # 2) If keys exhausted this round (or only one key), switch model bucket
                if (not switched) or keys_tried >= max(len(keys), 1):
                    nxt_model = _next_fallback_model(call_model)
                    if nxt_model and nxt_model not in models_tried:
                        call_model = nxt_model
                        models_tried.add(nxt_model)
                        switched = True
                        keys_tried = 0
                        logger.warning(f"[Groq] 429 → fallback model={call_model}")
                    elif nxt_model:
                        call_model = nxt_model
                        switched = True

                if call_model in _JSON_MODE_UNRELIABLE:
                    model_kwargs.pop("response_format", None)

                llm = _build_llm(
                    model=call_model,
                    api_key=call_key,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    model_kwargs=model_kwargs or None,
                )
                logger.warning(
                    f"[Groq] 429 attempt={attempt + 1} "
                    f"model={call_model} key={_mask_key(call_key)} switched={switched}"
                )
                time.sleep(
                    _parse_retry_wait_seconds(str(exc), attempt, switched=switched)
                )
        raise last_exc
    finally:
        _release_llm_lock(lock)


class Lanchain_models():
    def __init__(self):
        pass

    def get_chat_model(self, agent_name: str | None = None):
        model = get_groq_model_name(agent_name)
        max_tokens = get_agent_max_tokens(agent_name, json_mode=False)
        key = get_current_groq_api_key()
        kwargs = {
            "model": model,
            "temperature": 0.6,
            "max_tokens": max_tokens,
        }
        if key:
            kwargs["api_key"] = key
        return ChatGroq(**kwargs)

    def get_json_chat_model(self, agent_name: str = "aggregation"):
        """Aggregation model — JSON mode only on models that support it reliably."""
        model = get_groq_model_name(agent_name)
        max_tokens = get_agent_max_tokens(agent_name, json_mode=True)
        key = get_current_groq_api_key()
        kwargs = {
            "model": model,
            "temperature": 0.4,
            "max_tokens": max_tokens,
        }
        if model not in _JSON_MODE_UNRELIABLE:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        if key:
            kwargs["api_key"] = key
        return ChatGroq(**kwargs)
