"""This module handles the RAG (Retrieval-Augmented Generation) logic, including document retrieval, reasoning, and response generation."""

from pydantic_ai import Agent, RunContext
from pydantic_ai import AgentRunResult
from .database import KnowledgeBase
from .models import RAGResponse

        
class RAGAgent:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        
        # 1. We define the Agent with the "System prompt" and the "Output Format" -> We tell LLM it has to be a RAG assistant and use tools
        # We initialize the Agent inside __init__ so each RAGAgent is a self-contained unit, we "bind" that specific LLM configuration
        # to that specific instance of our class. It also allows to attach tools specifically to that agent
        self.agent = Agent(
            model="google-gla:gemini-2.5-pro",
            output_type = RAGResponse,
            system_prompt=(
                "You are a helpful assistant. Use the 'search_documents' tool to find"
                "information. Answer the user question clearly. You must provide "
                "the source snippet and source_file for every answer based on the tools. "
                "If the answer isn't in the context, say you don't know."
            )
        )
        
        # 2. We register a "Tool". The agent will call this automatically if it needs information
        # In pydantic-ai the @agent.tool decorator needs to know to which agent it belongs to. Since our agent
        # is an instance variable (self.agent), we define the tool inside __init__ so it can "hook" into self.agent immediatly
        @self.agent.tool
        async def search_documents(ctx: RunContext[None], query: str) -> str: # ctx: RunContext is required by pydantic-ai
            """Search the knowledge base for relevant document chunks"""
            results = self.kb.query(query, top_k=3) # List of RetrievedDocument objects
            
            formatted_results = []
            # res is a RetrievedDocument object
            for res in results:
                formatted_results.append(
                    f"File: {res.metadata['title']}\nContent: {res.content}"
                )
            return "\n\n---\n\n".join(formatted_results)
    
    async def run(self, user_query: str) -> AgentRunResult[RAGResponse]:
        """The mainentry point to talk to the agent."""
        # The agent handles the 'Augmentation' and 'Generation' internally
        result = await self.agent.run(user_query)
        
        # result is a RunResult object, with result.data we can get the RAGResponse object
        # We return the whole result so the evaluator can acces usage metadata and tool logs
        return result
                

    # Legacy code -> this is standard RAG, not agentic RAG
    # The benefit of Agentic RAG is that the model not necesarrily will search in the database, it will do it
    # if it cannot answer by itself. For example, with the standard RAG, if the user asks "who are you", the system would
    # search in the vector DB and lose time, but with Agentic RAG the LLM will anser "I'm an assistant" and won't search in 
    # the databaes.

    #def augment_prompt_with_context(self, query: str) -> str:
    #    """Prompt augmentation searching informatrion in Vector DB"""
    #    # 1. Retrieve relevant documents
    #    search_results = self.knowledge_base.query(query=query, top_k=5)
    #    context_parts = []
    #    
    #    # Combined retrieved chunks into a single context block
    #    for i, result in enumerate(search_results, 1):
    #        context_parts.append(
    #            f"Source {i}: {result['metadata']['title']}\n{result['content']}"
    #        )
    #    
    #    # 2. Build context
    #    context = "\n\n".join(context_parts)
    #    
    #    # 2. Build final prompt = instructions + context + question
    #    augmented_prompt = f"""
    #    Based on the following context, answer the user's question.
    #    CONTEXT:
    #    {context}
    #    
    #    QUESTION: {query}
    #    
    #    Answer clearly and only using the provided information.
    #    """
    #    
    #    return augmented_prompt
    # 
    #def generate_answer(self, query: str) -> RAGResponse:
    #    """Main entry point to answer a user query using RAG (augmented prompt)"""
    #    augmented_prompt = augment_prompt_with_context(query)
    #    answer = self.agent.generate(augmented_prompt)
    #    # Returned structured response
    #    return RAGResponse(
    #        answer = answer
    #    )
            
           
            
            
            