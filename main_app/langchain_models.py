from langchain_groq import ChatGroq

class Lanchain_models():
    def __init__(self):
        pass

    def get_chat_model(self):
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.6,
            max_tokens=1500,
        )
        return llm

    def get_json_chat_model(self):
        """Aggregation model — JSON mode, capped at 900 tokens for the synthesis output."""
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.4,
            max_tokens=900,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        return llm