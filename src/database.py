"""In this script we define the VectorStore and KnowledgeBase classes."""

from typing import List, Dict, Any
import os
from pathlib import Path
import threading
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .models import RetrievedDocument
import logging


# Environment settings for performance -> Protect Database/Embedding processes
# Disable parallelism in Hugging Face tokenizers to prevent race conditions/deadlocks
# This tells Hugging face tokenizores "Do NOT use your internal multithreading"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class VectorStore:
    def __init__(self, collection_name: str, persist_directory: str = "./rag/vector_store"):
        self.collection_name = collection_name
        # PersistentClient so the data survives script restarts
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def reset(self) -> None:
        """Reset the vector store by deleting all documents."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception as e:
            logging.warning(f"Failed to delete collection '{self.collection_name}': {e}")
        finally:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        print(f"Vector store '{self.collection_name}' has been reset.")
        
        
# KnowledgeBase class: A knowledge base is an interface that manages knowledge (documents + embeddings + retrieval)
class KnowledgeBase:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        # We define embedder here so both ingest and query can use it
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def reset_assets(self):
        """Passes the reset command down to the VectorStore."""
        print("Resetting underlying VectorStore")
        self.vector_store.reset()
               
    def _chunk_text(self, documents: List[Dict[str, Any]]) -> List[Dict]:  
        """Split documents into smaller chunks for better retrieval."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500, 
            chunk_overlap = 50, 
            separators = ["\n\n", "\n", " ", ""]
        )
        # Chunk all documents
        chunks = []
        for doc in documents:
            doc_chunks = splitter.split_text(doc["content"])
            for i, chunk in enumerate(doc_chunks):
                chunks.append({
                    "id": f"{doc['id']}_chunk_{i}", # Unique chunk ID
                    "title": doc["title"],
                    "content": chunk,
                    "source_doc": doc["id"]
                })
        
        print(f"Created {len(chunks)} chunks from the documents.")
        return chunks
         
    def add_chunks_to_collection(self, chunks: List[Dict]):
        """Add documents chunks to the vector collection."""
        # Prepare data
        ids = [chunk["id"] for chunk in chunks]
        contents = [chunk["content"] for chunk in chunks]
        
        # Metadatas for filtering / debugging
        metadatas = [
            {
                "title": chunk["title"],
                "source": chunk["source_doc"]
            }
            for chunk in chunks
        ]
        
        # Embed documents before storing in the vector database
        embeddings = self.embedder.encode(contents).tolist()
        
        # Apsert chunks to collection using upsert (add if not existsts, update if exists) which is safer than add
        self.vector_store.collection.upsert(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        print(f"Stored {len(chunks)} chunks in the vector database.")
        
        return self.vector_store.collection
    
    def ingest_file(self, file_path: str | Path):
        """
        Read a single file and ingest it, that is, take a raw file and put its content into a system in a usable (searchable) form.
        In this case, which is a RAG system, we ingest into a KnowledgeBase object, which internally stores data in Vector database.
        We use as Vector database Chroma, but it could be FAISS or another.
        NOTE: We have a function to ingest multiple files at a time but we keep this in case we want to ingest a SINGLE file into the KnowledgeBase,
        ex: add or update one document, test ingestion, new files arrive incrementally.
        """
        file_path = Path(file_path)
        try:
            # Read file content
            text = file_path.read_text(encoding="utf-8").strip()
            if not text:
                print(f"Skipping empty file: {file_path}")
                return # This stops processing this file to avoid creating empty chunks
            
            # Build document structure expected by _chunk_text
            # Dict[str, Any] -> is a dictionary where keys are strings and values can be any type
            document: Dict[str, Any] = {
                "id": str(file_path), # Use full path for uniqueness
                "title": file_path.name, #file_path.name is the file name including extension, ex:  Path('/home/user/document.txt').stem -> 'document.txt'
                "content": text
            }
            # 1. Chunk document
            chunks = self._chunk_text([document])
            
            # 2. Store chunks
            self.add_chunks_to_collection(chunks)
            
            print(f"Ingested file: {file_path}")
        except Exception as e:
            print(f"Failed to ingest {file_path}")
            
    def ingest_folder_batch(self, folder_path: str):
        """Ingest all .md files from a folder (recursively, that is, including all subfolders)."""
        folder = Path(folder_path)
        
        documents: List[Dict[str, Any]] = []
        
        # 1. Read all files and build document list
        for file_path in folder.rglob("*.md"):  # recursive search
            try:
                # Read file content
                text = file_path.read_text(encoding="utf-8").strip()
                # Skip empty file, using continue instead of return to continue looping
                if not text:
                    print(f"Skipping empty file: {file_path}")
                    continue
                documents.append({
                    "id": str(file_path), # unique identifier
                    "title": file_path.name,
                    "content": text
                })
            except Exception as e:
                print(f"Failed to read file {file_path}: {e}")
        
        # If no valid documents found, stop execution of entire method
        if not documents:
            print(f"No valid documents found for ingestion")
            return
        
        # 2. Chunk all documents together (this is more efficient)
        chunks = self._chunk_text(documents)
        print(f"Created {len(chunks)} chunks")
        
        # 3. Store all chunks in one batch (embeddings + DB write)
        self.add_chunks_to_collection(chunks)
        print(f"Batch ingestion complete")
        
    def user_query_processing(self, query: str) -> str:
        """Clean and prepare the raw user input. Separate from embedding to allow for future logic like PII redaction or expansion."""
        # For simplicity, we just lowercase and strip the query here
        cleaned_query = query.lower().strip()
        return cleaned_query
    
    def query(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        """Query the vector store using processed text and return relevant documents."""
        processed_text = self.user_query_processing(query)
        query_embedding = self.embedder.encode([processed_text])[0]
        
        # Perform similarity search -> Search top_k similar chunks
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()], # Is a list of queries even if there's only one element
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        
        search_results = []
        
        # Iterate through results, we access [0] because we assume we only sent one query string
        for i in range(len(results["ids"][0])):
            # Access the distance
            distance = results["distances"][0][i]
            
            # Convert distance -> similarity (higher = better)
            similarity = 1 - distance
            
            # Package it into our formal object
            search_results.append(RetrievedDocument(
                id=results["ids"][0][i],
                score=similarity,
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i]
            ))
        
        return search_results


# ----- THREAD-SAFE INITIALIZATION ------

# NOTE: If we put the initialization inside the RAGAgent class, every time we create a new Agent we would reload the database
# and the model -> we could run out of memory. By doing it outside and using @lru_cache we ensure the database is loaded once, and
# every agent just "borrows" a reference to it.
    
# This lock is a "safety guard". It ensures only one thread runs the initialization code at a time.
init_lock = threading.Lock() 

# This cache remembers the result. If we call this function with the same folder/collection name twice, it returns the
# already loaded object instead of creaing a new one.

@lru_cache
def _get_kb_instance(collection_name: str, persist_dir: str) -> KnowledgeBase:
    """Internal (private) function: It actually creates the KB object"""
    # Note: If the vector store was already created then we are not creating a new one but getting the existing
    vs = VectorStore(collection_name=collection_name, persist_directory=persist_dir)
    return KnowledgeBase(vector_store=vs) 


def get_knowledge_base(collection_name: str, persist_dir: str) -> KnowledgeBase:
    """Public Entry Point: This is "thread-safe" because of the 'with' block. If 10 threads
    call this, they will wait in line for the lock, ensuring the Database diles are opened safely"""  
    with init_lock:
        return _get_kb_instance(collection_name, persist_dir) 
                   

 

        
        
        
    
    
    
    