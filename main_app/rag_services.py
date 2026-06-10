import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .models import *
from Ai_strategy_engine.logger import logger
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from .helper import HashService, ChunkService
from .embedding_service import Embedding
import os

class Rag_service():
    def __init__(self, thread_id):
        self.thread_id = thread_id
        self.hash_service = HashService()
        self.chunk_service = ChunkService()

    def load_documents(self, documents):
        logger.info(f"Loading documents started for thread {self.thread_id}")

        all_docs = []

        for doc in documents:
            try:
                file_path = doc.document_path.path
                extension = os.path.splitext(file_path)[1].lower()

                logger.info(
                    f"Loading document: {file_path} "
                    f"(extension={extension})"
                )

                if extension == ".pdf":
                    loader = PyPDFLoader(file_path)

                elif extension in [".docx", ".doc"]:
                    loader = Docx2txtLoader(file_path)

                elif extension in [".txt", ".md"]:
                    loader = TextLoader(
                        file_path,
                        encoding="utf-8"
                    )

                else:
                    logger.warning(
                        f"Unsupported file type: {file_path}"
                    )
                    continue

                loaded_docs = loader.load()

                # Add document metadata
                for loaded_doc in loaded_docs:
                    loaded_doc.metadata["document_id"] = doc.id
                    loaded_doc.metadata["file_name"] = os.path.basename(
                        file_path
                    )

                all_docs.extend(loaded_docs)

                logger.info(
                    f"Successfully loaded {file_path}"
                )

            except Exception as e:
                logger.exception(
                    f"Failed to load {file_path}: {str(e)}"
                )
                continue

        if not all_docs:
            logger.warning(
                f"No valid documents loaded for thread {self.thread_id}"
            )
            return False, None

        logger.info(
            f"Successfully loaded {len(all_docs)} document chunks "
            f"for thread {self.thread_id}"
        )

        return True, all_docs   
    def split_documents(self, documents):
        logger.info(f"Split documents start for thread {self.thread_id}")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)
            # 🔥 ADD THIS PART
            for i, doc in enumerate(splits):
                content = doc.page_content
                doc.metadata["chunk_hash"] = self.hash_service.create_hash_each_chunk(content)
                doc.metadata["thread_id"] = str(self.thread_id)
                doc.metadata["position"] = i  # optional but useful

            logger.info(f"Split documents success for thread {self.thread_id}")
            return True, splits
        except Exception as e:
            logger.error(e)
            return False, None
        
        
    def filter_new_chunks(self, splits):
        logger.info(f"Filtering new chunks for thread {self.thread_id}")

        existing_hashes = set(
            Chunk.objects.filter(thread_id=self.thread_id)
            .values_list("chunk_hash", flat=True)
        )

        new_chunks = []
        for split in splits:
            if split.metadata["chunk_hash"] not in existing_hashes:
                new_chunks.append(split)
                self.chunk_service.create_chunk(self.thread_id, split.metadata["chunk_hash"], split.page_content)

        logger.info(f"New chunks count: {len(new_chunks)}")
        return new_chunks
    
    def create_and_store_vectors_in_faiss(self, splits):
        logger.info(f"Create and store vectors in faiss start for thread {self.thread_id}")
        try:
            if not splits:
                logger.info("No new chunks to embed")
                return True, None, False

            embedding_service = Embedding()
            document_embeddings = embedding_service.create_embeddings(
                [doc.page_content for doc in splits]
            )
            # 4️⃣ Convert to numpy float32
            vectors = np.array(document_embeddings).astype("float32")
            # 5️⃣ Get dimension
            dimension = vectors.shape[1]
            # 6️⃣ Create HNSW index
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
            # 7️⃣ Add vectors to index
            index.add(vectors)
            logger.info(f"HNSW index created successfully for thread {self.thread_id}")
            return True, index ,True

        except Exception as e:
            logger.exception(
                f"Faiss HNSW creation failed for thread {self.thread_id}"
            )
            return False, None ,False
        
    def get_documents_exist_in_db(self, documents):
        logger.info(f"Get documents exist in db start for thread {self.thread_id}")
        try:
            documents_exist = Document.objects.filter(thread_id=self.thread_id)
            if documents_exist.exists():
                logger.info(f"Documents exist for thread {self.thread_id}")
                return True
            logger.info(f"Documents do not exist for thread {self.thread_id}")
            return False
        except Exception as e:
            logger.exception(f"Error getting documents exist in db: {str(e)}")
            return False
