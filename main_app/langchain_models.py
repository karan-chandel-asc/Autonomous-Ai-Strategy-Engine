import os
from langchain_groq import ChatGroq

# Groq retired meta-llama/llama-4-scout-17b-16e-instruct on 2026-07-17.
# Official replacements: openai/gpt-oss-120b or qwen/qwen3.6-27b
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def get_groq_model_name() -> str:
    return os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)


class Lanchain_models():
    def __init__(self):
        pass

    def get_chat_model(self):
        llm = ChatGroq(
            model=get_groq_model_name(),
            temperature=0.6,
            max_tokens=1500,
        )
        return llm

    def get_json_chat_model(self):
        """Aggregation model — JSON mode, capped at 900 tokens for the synthesis output."""
        llm = ChatGroq(
            model=get_groq_model_name(),
            temperature=0.4,
            max_tokens=900,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        return llm
