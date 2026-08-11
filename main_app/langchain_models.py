import os
import re
import time
from langchain_groq import ChatGroq

# Groq retired meta-llama/llama-4-scout-17b-16e-instruct on 2026-07-17.
# Note: llama-3.3-70b-versatile is scheduled for shutdown on 2026-08-16.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Free / on_demand tiers are TPM-capped (~12k). Keep completions short.
# No-tools path ≈ 8 LLM calls/report; these caps keep a full run under 12k TPM.
_CHAT_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "450"))
_JSON_MAX_TOKENS = int(os.environ.get("GROQ_JSON_MAX_TOKENS", "280"))
_RATE_LIMIT_ATTEMPTS = int(os.environ.get("GROQ_RATE_LIMIT_ATTEMPTS", "5"))


def get_groq_model_name() -> str:
    return os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def invoke_with_rate_limit_retry(llm, messages, attempts: int = _RATE_LIMIT_ATTEMPTS):
    """Invoke Groq LLM; on 429 TPM errors, sleep for the suggested wait and retry."""
    last_exc = None
    for attempt in range(attempts):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            is_rate = (
                "rate_limit" in msg.lower()
                or "429" in msg
                or type(exc).__name__ == "RateLimitError"
            )
            if not is_rate or attempt == attempts - 1:
                raise
            match = re.search(r"try again in ([\d.]+)\s*s", msg, re.I)
            wait = float(match.group(1)) + 0.75 if match else min(2 ** attempt + 2, 30)
            time.sleep(wait)
    raise last_exc


class Lanchain_models():
    def __init__(self):
        pass

    def get_chat_model(self):
        llm = ChatGroq(
            model=get_groq_model_name(),
            temperature=0.6,
            max_tokens=_CHAT_MAX_TOKENS,
        )
        return llm

    def get_json_chat_model(self):
        """Aggregation model — JSON mode, capped for free-tier TPM."""
        llm = ChatGroq(
            model=get_groq_model_name(),
            temperature=0.4,
            max_tokens=_JSON_MAX_TOKENS,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        return llm
