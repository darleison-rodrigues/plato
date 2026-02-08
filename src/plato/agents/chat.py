import logging
from typing import List, Dict, Any
from .base import BaseAgent
# We assume the user has a Pipeline instance or we create a minimal one for RAG
from plato.core import Pipeline 

logger = logging.getLogger(__name__)

class ChatAgent(BaseAgent):
    """
    The 'Research Partner': Interactive RAG agent.
    """
    
    def __init__(self):
        super().__init__(model_override="llama3.2:3b")
        # We need the vector store access
        self.pipeline = Pipeline() 

    def run(self, query: str) -> str:
        """
        Answers a single question using RAG.
        """
        if not query.strip():
            return "Please ask a question."
            
        # 1. Retrieve Context
        try:
            results = self.pipeline.search_documents(query, n_results=5)
            # Combine relevant chunks
            chunks = []
            sources = set()
            
            combined = results.get("combined_results", [])
            for item in combined:
                if item.get('type') == 'document':
                    content = item.get('content', '')
                    src = item.get('source_document', 'Unknown')
                    chunks.append(f"Source: {src}\n{content}")
                    sources.add(src)
            
            context_text = "\n\n---\n\n".join(chunks)
            if not context_text:
                return "I couldn't find any relevant information in your library."

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return "Sorry, I had trouble searching your library."

        # 2. Generate Answer
        prompt = f"""
        You are PLATO, a helpful research assistant.
        Answer the user's question based ONLY on the context provided below.
        If the context doesn't contain the answer, say "I don't know based on these documents."

        CONTEXT:
        {context_text}

        USER QUESTION:
        {query}
        
        ANSWER:
        """
        
        # Stream the response if possible, but for simplicity we block here
        # (Rich handles streaming well, but let's keep it simple first)
        response = ""
        if self.client:
             stream = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=True
            )
             # We return a generator or full string?
             # For CLI integration, full string is easier to start
             for chunk in stream:
                 response += chunk['response']
        else:
            response = "Error: Client not connected."
            
        return response
