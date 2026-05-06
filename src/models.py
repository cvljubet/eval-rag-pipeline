"""In this script we define 'contracts', it includes:
- RetrievedDocument: The structure of the retrieved chunk (even if we call it Document)
 - RagResponse: The structure for Agent's answers
 - JudgeOutput: The structure for Evaluator's grades (calificaciones)"""
 
# from typing import Literal, List
from typing import Dict, Any
from pydantic import BaseModel, Field


# Define RetrievedDocument to use later for Type hinting and better code clarity. This will represent the structure of documents retrieved from the vector store.
class RetrievedDocument(BaseModel):
    id:str # unique identifier for the document
    score:float # relevance score from the retrieval process - NOTE: This is retrieval relevance, NOT answer relevance, it comes from ChromaDB, is NOT the Judge LLM I'm using in models.py
    content:str # the actual text content of the document
    metadata:Dict[str, Any] # additional information about the document (e.g., title, category, source)


class RAGResponse(BaseModel):
    """The structured output returned by the main RAG agent"""
    answer: str = Field(description="The generated response from the RAG system")
    source_snippet: str = Field(description="The exact quote retrieved from the vector database used to generate the answer")
    source_file: str = Field(description="Filename of the source document (the document from which the chunk was created)")
    # The field confidence would enforce the "self-reflect" before finishing the answer, it looks at the source snippet and thinks:
    # "Does this text actually answer the question clearly? If yes, I'll return 'high'" But is subjective, is the agent opinion
    # on its own work, I'll keep it comment in case I want to test it later
    #confidence: Literal["high", "medium", "low"] = Field(
    #    description="Confidence level: 'high' if answer is directly from docs, 'medium' if inferred, 'low' if uncertain"
    #    )


class JudgeOutput(BaseModel):
    """The structured scorecard filled out bu the LLM Judge during evaluation, we need this because we want to make
    sure the LLM returns a structured output with the evaluations"""
    is_faithful: bool = Field(description="True if the answer is supported by the source (derived solely from the provided source snippet)")
    faithfulness_score: float = Field(description="Score from 0 to 1, measures if the generated answer is derived solely from the provided source snippet")
    relevance_score: float = Field(description="Score from 0 to 1, measures how useful is the retrieved document, is it answering the query?")
    reasoning: str = Field(description="Explanation for the assigned scores")
    


    
    
