"""
Streamlit app for the RAG system
This script serves as the frontend for the Validated RAG pipeline.
"""

import streamlit as st
import asyncio
from src.database import get_knowledge_base
from src.agent import RAGAgent
from evals.judge import RAGEvaluator

# --- 1. CONFIGURATION ---
# We keep these separate so they are easy to change if your folder structure moves
PERSIST_DIR = "./vector_store"
COLLECTION_NAME = "images_rag"

def initialize_session():
    """
    Streamlit runs from top-to-bottom on every interaction.
    'st.session_state' is a dictionary that persists data so we don't 
    reload the heavy database or models every time a user clicks a button.
    """
    if "kb" not in st.session_state:
        # get_knowledge_base connects to ChromaDB. We only want to do this once.
        st.session_state.kb = get_knowledge_base(
            collection_name=COLLECTION_NAME, persist_dir=PERSIST_DIR
        )
    
    if "agent" not in st.session_state:
        # We store the Agent class here so it's ready to go.
        st.session_state.agent = RAGAgent(knowledge_base=st.session_state.kb)
    
    if "evaluator" not in st.session_state:
        # The Judge agent is also stored in the session state.
        st.session_state.evaluator = RAGEvaluator()

async def run_rag_query(query: str):
    """
    This is our core logic bridge. It is 'async' because our Agent 
    and Judge use asynchronous API calls to Gemini.
    """
    # 1. Ask the RAG Agent (Retrieval + Generation)
    agent_result = await st.session_state.agent.run(query)
    
    # 2. Ask the Judge Agent (Validation)
    evaluation = await st.session_state.evaluator.evaluate(query, agent_result)
    
    # We return a dictionary so it's easy to display in the UI
    return {
        "query": query,
        "answer": agent_result.output.answer,
        "source_snippet": agent_result.output.source_snippet,
        "source_file": agent_result.output.source_file,
        "evaluation": evaluation 
    }

def main():
    # --- 2. UI SETUP ---
    # set_page_config must be the very first Streamlit command used.
    st.set_page_config(page_title="Validated RAG System", layout="wide")
    st.title("🔬 RAG: Personal Knowledge Retriever")
    
    st.markdown("""
    This system retrieves context from local documents and uses a secondary 
    **LLM-Judge** to audit the answer for hallucinations.
    """)
    
    # --- 3. INITIALIZATION ---
    try:
        initialize_session()
    except Exception as e:
        # If ChromaDB isn't found, we catch the error here and show a helpful message.
        st.error(f"Error loading knowledge base: {str(e)}")
        st.info("💡 Make sure you have run `python -m ingest` first!")
        return
    
    # --- 4. USER INPUT ---
    # st.text_input returns a string. It triggers a rerun when 'Enter' is pressed.
    user_query = st.text_input("Ask a Question:", placeholder="e.g., Explain brightfield vs fluorescence...")
    
    # st.button returns True only in the exact moment it is clicked.
    if st.button("Get Answer", type="primary"):
        # st.spinner shows a loading animation while the code inside the 'with' block runs.
        with st.spinner("Searching documents and auditing answer..."):
            try:
                # IMPORTANT: Streamlit is synchronous. Since our pipeline is async,
                # we use asyncio.run() to create a bridge between the two.
                result = asyncio.run(run_rag_query(user_query))
                
                # We split the screen into two columns for a professional look.
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader("📝 Answer")
                    st.write(result["answer"])
                    
                    # st.info creates a nice blue box for the retrieved text.
                    st.subheader("📚 Source Context")
                    st.info(result["source_snippet"])
                    st.caption(f"Source file: {result['source_file']}")
                
                with col2:
                    st.subheader("⚖️ Audit")
                    eval_data = result["evaluation"]
                    
                    # st.metric is a specialized UI component for displaying scores.
                    st.metric("Faithfulness", f"{eval_data['score']}/1.0")
                    
                    # Color coding the verdict
                    if eval_data['score'] >= 0.7:
                        st.success("✅ Verified Accurate")
                    else:
                        st.warning("⚠️ Potential Hallucination")
                        
                    st.write("**Judge Reasoning:**")
                    st.write(eval_data['summary'])
            
            except Exception as e:
                st.error(f"❌ System Error: {str(e)}")

    # --- 5. SIDEBAR ---
    # st.sidebar puts elements in the left-hand drawer.
    with st.sidebar:
        st.subheader("ℹ️ System Architecture")
        st.write("- **Vector DB:** ChromaDB")
        st.write("- **Model:** Gemini 2.5 Pro")
        st.divider()
        st.markdown("Developed by: [Your Name]")

if __name__ == "__main__":
    main()