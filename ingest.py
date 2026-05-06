"""Script to get or create KnowledgeBase and ingest data.
We should run it once to create and populate our database, before running main.py.
Run this script from the root folder: python -m ingest"""

import os
from src.database import get_knowledge_base

# Absolute paths or paths relative to the root where we run the script
DATA_FOLDER = "./data"
PERSIST_DIR = "./vector_store"
COLLECTION_NAME = "images_rag"

def run_ingestion(data_folder: str, persist_dir: str, collection_name: str):
    # 1. Initialize the KB safely
    # We use the thread-safe function even if we have only 1 thread
    kb = get_knowledge_base(collection_name=collection_name, persist_dir=persist_dir)

    # 2. Optional: Reset the database if we want to start a new one
    # kb.reset_assets()
    
    # 3. Perform ingestion
    print(f"Starting ingestion from: {data_folder}")
    if os.path.exists(data_folder):
        kb.ingest_folder_batch(data_folder)
        print("Ingestion complete =D")
    else:
        print(f"Error: Folder {data_folder} not found")
        
        
if __name__ == "__main__":
    run_ingestion(
        data_folder=DATA_FOLDER, 
        persist_dir=PERSIST_DIR, 
        collection_name=COLLECTION_NAME
    )


