

from .rag_services import Rag_service
from .helper import HashService,DocumentService,faissService
from .models import *
from Ai_strategy_engine.logger import logger
class processDocuments:
    def __init__(self, thread_id):
        self.thread_id = thread_id
        self.rag_service = Rag_service(thread_id)
        self.hash_service = HashService()
        self.document_service = DocumentService()
    
    def process(self, documents):
        saved, message, document_ids, duplicate_files = self.document_service.save_document(self.thread_id, documents)

        if not saved:
            # All files were duplicates — check if existing docs are in DB for this thread
            logger.error(f"No documents available for thread {self.thread_id}: {message}")
            return False, message, duplicate_files
        else:
            existing = self.document_service.get_document(self.thread_id)

        success, loaded_docs = self.rag_service.load_documents(existing)
        if not success:
            logger.error(f"Loading documents failed for thread {self.thread_id}")
            return False, "Failed to load documents.", duplicate_files

        success, splits = self.rag_service.split_documents(loaded_docs)
        if not success:
            logger.error(f"Split documents failed for thread {self.thread_id}")
            return False, "Failed to split documents.", duplicate_files

        success, vector_index, new_embedding = self.rag_service.create_and_store_vectors_in_faiss(splits)
        if not success:
            logger.error(f"Create and store vectors in faiss failed for thread {self.thread_id}")
            return False, "Failed to create vectors.", duplicate_files

        if new_embedding:
            faiss_service = faissService()
            file_path = faiss_service.save_faiss_index_file_path(self.thread_id, vector_index)
            faiss_service.save_faiss_index_path_in_db(self.thread_id, file_path)

        return True, message, duplicate_files
        