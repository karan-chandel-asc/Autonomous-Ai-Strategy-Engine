import uuid
import hashlib
from .models import Thread,Vector_storage,Document,Chunk
import os
from .pydantic_schemas import InputValidation
from pydantic import ValidationError
from django.core.files.storage import default_storage
from django.conf import settings

class ThreadService:
    def generate_thread_id(self):
        return f"Thread_{str(uuid.uuid4())}"

    def get_thread(self, thread_id):
        try:
            return Thread.objects.get(id=thread_id)
        except Thread.DoesNotExist:
            return None
    
    def create_thread(self,owner, thread_id):
        thread =Thread.objects.create(owner=owner,thread_id=thread_id)
        return  thread.thread_id
    



class HashService:
    
    def create_hash_each_chunk(self, text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_uploaded_file_hash(self, uploaded_file):
        sha256 = hashlib.sha256()

        for chunk in uploaded_file.chunks():
            sha256.update(chunk)

        uploaded_file.seek(0)  # Reset pointer

        return sha256.hexdigest()
    
class DocumentService:
    def __init__(self):
        self.hash_service = HashService()

    def get_document(self, thread_id):
        try:
            return Document.objects.get(thread_id=thread_id)
        except Document.DoesNotExist:
            return None

    def create_document(self, thread_id, document_path, document_hash):
        document = Document.objects.create(
            thread_id=thread_id,
            document_path=document_path,
            document_hash=document_hash
        )
        return document.id

    def save_document(self, thread_id, documents):
        created_document_ids = []
        duplicate_files = []

        for document in documents:
            document_hash = self.hash_service.get_uploaded_file_hash(document)

            if self.check_document_already_exists(thread_id, document_hash):
                duplicate_files.append(document.name)
                continue

            file_path = f"uploads/{thread_id}/{document.name}"
            saved_path = default_storage.save(file_path, document)

            document_id = self.create_document(
                thread_id,
                saved_path,
                document_hash
            )

            created_document_ids.append(document_id)

        if created_document_ids:
            return (True, "Document(s) saved successfully.", created_document_ids, duplicate_files)

        return (False, "All uploaded documents already exist.", [], duplicate_files)

    def check_document_already_exists(self, thread_id, document_hash):
        return Document.objects.filter(
            thread_id=thread_id,
            document_hash=document_hash
        ).exists()

        
class ChunkService:
    def get_chunk(self, thread_id):
        try:
            return Chunk.objects.get(thread=thread_id)
        except Chunk.DoesNotExist:
            return None
    
    def create_chunk(self, thread_id, chunk_hash, content):
        return Chunk.objects.create(thread_id=thread_id, chunk_hash=chunk_hash, content=content)
    
    
    
class faissService:
    def save_faiss_index_file_path(self, thread_id, index):
        folder = "faiss_indexes"
        os.makedirs(folder, exist_ok=True)

        path = f"{folder}/thread_{thread_id}.index"
        faiss.write_index(index, path)
        return path
    
    def save_faiss_index_path_in_db(self, thread_id, index_path):
        try:
            thread = Thread.objects.get(thread_id=thread_id)
        except Thread.DoesNotExist:
            raise ValueError(f"Thread '{thread_id}' not found in DB")
        Vector_storage.objects.update_or_create(
            thread=thread,
            defaults={"vector_path": index_path, "metadata": {}}
        )
        
# Gemini query embedder — cached once per worker process
_GEMINI_EMB_CACHE = None


class HybridRAGService:
    """Hybrid keyword + semantic retrieval via Pinecone."""

    def __init__(self, document_ids: list):
        self.document_ids = [str(d) for d in document_ids]

    def retrieve(self, query: str, top_k: int = 2) -> tuple:
        """Return (context_str, kb_citations_list).

        context_str has inline [KB: filename, p.N] tags appended to each chunk.
        kb_citations_list is [{"doc_name": str, "page": int, "snippet": str}].
        """
        global _GEMINI_EMB_CACHE
        from .embedding_service import GeminiQueryEmbedding
        from .pinecone_service import PineconeService

        if not self.document_ids:
            return "", []

        if _GEMINI_EMB_CACHE is None:
            _GEMINI_EMB_CACHE = GeminiQueryEmbedding()

        try:
            query_vec = _GEMINI_EMB_CACHE.embed_query(query)
            results = PineconeService().query(
                self.document_ids, query_vec, top_k=3
            )
            if not results:
                return "", []

            # ── Keyword re-rank on the returned semantic results ───────
            query_words = set(query.lower().split())
            scored = []
            for r in results:
                kw = len(query_words & set(r["text"].lower().split())) / (len(query_words) or 1)
                scored.append((r["score"] + kw, r))
            scored.sort(reverse=True, key=lambda x: x[0])

            parts = []
            kb_citations = []
            seen = set()   # deduplicate by (doc_name, page)

            for _, item in scored[:top_k]:
                text = item["text"].strip()
                if not text:
                    continue
                doc_name = item.get("doc_name", "")
                page     = item.get("page", 0)
                if doc_name:
                    parts.append(f"{text}\n[KB: {doc_name}, p.{page}]")
                    key = (doc_name, page)
                    if key not in seen:
                        seen.add(key)
                        kb_citations.append({
                            "doc_name": doc_name,
                            "page":     page,
                            "snippet":  text[:120] + "…" if len(text) > 120 else text,
                        })
                else:
                    parts.append(text)

            return "\n\n---\n\n".join(parts), kb_citations

        except Exception as exc:
            from Ai_strategy_engine.logger import logger
            logger.warning(f"[HybridRAG] retrieval failed: {exc}")
            return "", []


class InputValidationService:
    def validate_input(self, data, files=None):
        try:
            # Exclude 'documents' from request.data — files are iterable by line,
            # so passing an InMemoryUploadedFile to List[Any] gives 90 lines not 1 file.
            cleaned_data = {
                key: value[0] if isinstance(value, list) else value
                for key, value in data.items()
                if key != 'documents'
            }
            # Inject proper Django file objects separately
            if files is not None:
                cleaned_data['documents'] = files

            validated_data = InputValidation(**cleaned_data)
            return True, validated_data

        except ValidationError as e:
            first = e.errors()[0]
            message = first["msg"].replace("Value error, ", "")
            return False, message