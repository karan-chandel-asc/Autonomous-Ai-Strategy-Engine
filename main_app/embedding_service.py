import os
import cohere


class Embedding:
    """Cohere embed-english-v3.0 — used for document indexing (1024-dim)."""

    def __init__(self):
        self.client = cohere.Client(os.environ.get("cohre_embedding_api_key", ""))

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document",
        )
        return response.embeddings


class CohereQueryEmbedding:
    """Query embedding using Cohere embed-english-v3.0 (1024-dim)."""

    def __init__(self):
        self.client = cohere.Client(os.environ.get("cohre_embedding_api_key", ""))

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_query",
        )
        return response.embeddings[0]
