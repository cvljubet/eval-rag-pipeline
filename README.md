# RAG: The Personal Knowledge Retriever
This project is part of an AI Upskilling course

## Overview

Build a functional, single-agent RAG (Retrieval-Augmented Generation) application. This serves as a foundational "sprint" to prepare you for the “Second Brain” Capstone by mastering Pydantic AI agents and tool-based context retrieval.

## Requirements

Design a single-agent system that can answer questions based on a specific local dataset (such as a meeting transcript or technical documentation). The agent must retrieve relevant context before generating a response to ensure accuracy.

The system should demonstrate the move from a basic "knowledge-less" chatbot to one that uses external data to provide grounded answers.

## In Scope
- **Use Pydantic AI**: Define a specialized agent with a clear system prompt.
- **Implement RAG**: Create a tool (function) that the agent can call to search and retrieve relevant context from a vector database.
- **Structured Outputs**: Use Pydantic models to ensure the agent returns both the final answer and the source_snippet used.
- **Vector Database**: Create embeddings from text documents, store them in a vector database, and implement a retriever pipeline to fetch relevant context for LLM responses.
- **Local Data**: Use a single .txt or .md file as your knowledge base.
- **Evals**: Use LLMJudge for evaluations. Additionally, as a stretch goal, explore how to compute a faithfulness metric.

## Prerequisites
- Python 3.12 or higher
- A Google Gemini API key

## Setup Instructions

### 1. Clone and create virtual environment
```bash
git clone <your-repo>
cd rag
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get your Google Gemini API Key

- Go to [Google AI Studio](https://ai.google.dev)
- Create an API key
- Set the environment variable:

```bash
export GEMINI_API_KEY=your_key_here
```

Or permanently add it to `~/.zshrc`:

```bash
echo 'export GEMINI_API_KEY=your_key_here' >> ~/.zshrc
source ~/.zshrc
```

## How to Run

### Prerequisites for Both Methods
Before running the system (terminal or Streamlit), you must first ingest the knowledge base:

```bash
python -m ingest
```

This processes your data files and creates the vector database in `./vector_store`. You only need to run this once.

### Option 1: Terminal (Batch Processing)

Run the RAG pipeline with predefined queries:

```bash
python -m main
```

This executes all sample queries, evaluates each response with the judge, and prints the results.

### Option 2: Streamlit App (Interactive)

Run the interactive web application:

```bash
python -m streamlit run app.py
```

Then open your browser to `http://localhost:8501`. You can:
- Ask custom questions in real-time
- See answers with source snippets
- View evaluation scores (faithfulness and relevance)
- Explore the judge's reasoning

## Project Structure

```
rag/
├── app.py                 # Streamlit interactive application
├── main.py                # Terminal entry point (batch processing)
├── ingest.py              # Data ingestion and vector store creation
├── requirements.txt       # Python dependencies
├── data/                  # Knowledge base documents
│   ├── brightfield_images.md
│   └── fluorescence_images.md
├── vector_store/          # ChromaDB vector database (created by ingest.py)
├── src/
│   ├── agent.py          # RAG Agent with tool definition
│   ├── database.py       # Vector store interface
│   └── models.py         # Pydantic data models
└── evals/
    └── judge.py          # LLM Judge for evaluation
    
1. Knowledge Ingestion & Vector Storage
Data Processing: Scientific documentation (PDFs/Text) is processed and split into semantic chunks. This data was extracted from Wikipedia, you can create your own documents with information of your interest and try different queries.

Embeddings: We generate high-dimensional vector representations of these chunks using sentence-transformers.

Vector Store: These embeddings are stored in ChromaDB, allowing for lightning-fast similarity searches when a user asks a question.

2. The RAG Pipeline (The Agent)
Retrieval: When a query is received, the system performs a vector search to find the most relevant context snippets.

Augmentation: The user’s question and the retrieved snippets are bundled into a structured prompt.

Generation: An LLM (Gemini 2.5 Pro) generates a response. To ensure reliability, the output is strictly typed using Pydantic models, returning both the answer and the exact source_snippet used.

3. Automated Auditing (The Judge)
This is the core "Quality Control" layer of the project:

The Agent’s output is passed to a second, independent instance of the LLM.

Validation: The Judge compares the answer against the source_snippet and the original query.

Scoring: It fills out a "Report Card" (JudgeOutput) containing:

Faithfulness: Does the answer stick to the facts in the source? (Prevents Hallucinations)

Relevance: Does the answer actually solve the user's problem?

Reasoning: A natural language explanation of why the score was given.

4. Asynchronous Execution
The entire system is built on Python’s asyncio framework.

This ensures that while the system is waiting for an API response from Gemini or a database read from ChromaDB, the CPU is unblocked and ready to handle other tasks, making the pipeline **highly scalable**.

## Models Used

- **Model**: `google-gla:gemini-2.5-pro`
- **Requires**: `GEMINI_API_KEY` environment variable
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB






