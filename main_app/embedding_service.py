import os
from langchain_community.embeddings import HuggingFaceEmbeddings as _HFE


class Embedding:
    """HuggingFace BAAI/bge-small-en-v1.5 — used for document indexing only (384-dim)."""

    def __init__(self):
        self.model = _HFE(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def create_embeddings(self, texts: list[str]) -> list[float]:
        return self.model.embed_documents(texts)


class GeminiQueryEmbedding:
    """Query embedding — tries Gemini gemini-embedding-001 first, falls back to HuggingFace.

    Gemini requires a valid AI Studio key (AIzaSy...) from https://aistudio.google.com/apikey
    output_dimensionality=384 keeps vectors compatible with the existing Pinecone index.
    """

    def __init__(self):
        self._gemini = None
        self._hf = None
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self._gemini = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=api_key,
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=384,
                )
                # Warm-up probe to detect invalid key early
                self._gemini.embed_query("test")
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"[GeminiQueryEmbedding] Gemini unavailable ({e}), falling back to HuggingFace"
                )
                self._gemini = None

        if self._gemini is None:
            self._hf = _HFE(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

    def embed_query(self, text: str) -> list[float]:
        if self._gemini is not None:
            return self._gemini.embed_query(text)
        return self._hf.embed_query(text)
