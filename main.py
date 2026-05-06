"""Main file to run RAG pipeline.
Before running this script make sure ingest.py was previously executed to create the KnowledgeBase
Run this script from the root folder: python -m main"""

import asyncio
from typing import List
from src.database import get_knowledge_base
from src.agent import RAGAgent
from src.models import RAGResponse
from evals.judge import RAGEvaluator

# Absolute paths or paths relative to the root where we run the script
PERSIST_DIR = "./vector_store"
COLLECTION_NAME = "images_rag"

queries = [
    "What is my favorite color?",
    "Where is located Tokyo?",
    "Tell me an example where brightfield images are useful",
    "What is the difference between brightfield and fluorescence images?" 
]

# Async def because we use 'await' inside
async def run_rag_agent(collection_name: str, persist_dir: str, queries: List[str]) -> RAGResponse:

    # 1. Load knowledge base
    kb = get_knowledge_base(collection_name=collection_name, persist_dir=persist_dir)

    # 2. Instantiate a RAGAgent object
    rag_agent = RAGAgent(knowledge_base=kb)
    
    # 3. Create one evaluator instance
    evaluator = RAGEvaluator()

    # 3. Ask the agent and evaluate
    results = []
    for query in queries:
        agent_run_result = await rag_agent.run(query)
        
        # 4. Evaluate the response
        evaluation = await evaluator.evaluate(query, agent_run_result) # await the judge
        
        results.append({
            "query": query,
            "answer": agent_run_result.output.answer,
            "source": agent_run_result.output.source_snippet,
            "evaluation": evaluation['summary']   
        })
    
    # 5. Final printout
    print("\n"+"="*30)
    for res in results:
        print(f"Q: {res['query']}")
        print(f"A: {res['answer']}")
        print(f"Source: {res['source']}")
        print(f"Judge: {res['evaluation']}\n")

   
    return results


if __name__ == "__main__":
    asyncio.run(run_rag_agent(collection_name=COLLECTION_NAME, persist_dir=PERSIST_DIR, queries=queries))