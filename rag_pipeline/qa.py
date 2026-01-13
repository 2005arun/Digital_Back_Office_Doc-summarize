"""
Question Answering Module
RAG-based Q&A using retrieved context and LLM.
"""

import os
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from .retriever import Retriever


# Load environment variables
load_dotenv()


class QAEngine:
    """RAG-based question answering engine."""
    
    # Fallback message when knowledge base doesn't contain the information
    NO_INFO_MESSAGE = "The knowledge base does not contain this information."
    
    def __init__(self, 
                 retriever: Retriever = None,
                 llm_provider: str = "groq",
                 data_dir: str = "data"):
        """
        Initialize the QA engine.
        
        Args:
            retriever: Retriever instance (creates new if None)
            llm_provider: LLM provider ('groq' or 'openai')
            data_dir: Directory for data storage
        """
        self.retriever = retriever or Retriever(data_dir=data_dir)
        self.llm_provider = llm_provider
        self.client = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize the LLM client."""
        if self.llm_provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                from groq import Groq
                self.client = Groq(api_key=api_key)
                self.model = "llama-3.1-8b-instant"  # Fast and free
            else:
                print("⚠️ GROQ_API_KEY not found. Set it in .env file.")
        elif self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                self.model = "gpt-3.5-turbo"
            else:
                print("⚠️ OPENAI_API_KEY not found. Set it in .env file.")
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            LLM response text
        """
        if self.client is None:
            return "⚠️ LLM not configured. Please set API key in .env file."
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ LLM error: {str(e)}"
    
    def answer(self, 
               question: str, 
               top_k: int = 3,
               include_sources: bool = True) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            include_sources: Whether to include sources in response
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(question, top_k=top_k)
        
        # Check if we have relevant context
        if not chunks:
            return {
                'question': question,
                'answer': self.NO_INFO_MESSAGE,
                'sources': [],
                'has_context': False
            }
        
        # Format context
        context = self.retriever.format_context(chunks, include_sources=True)
        
        # Build prompt
        system_prompt = """You are a helpful technical documentation assistant. 
Answer questions ONLY using the provided documentation context.
If the context doesn't contain enough information to answer the question, say so.
Be concise and accurate. Include code examples when relevant."""
        
        prompt = f"""Based on the following documentation:

{context}

Question: {question}

Provide a clear, accurate answer based ONLY on the documentation above. 
If the documentation doesn't contain the answer, say "The knowledge base does not contain this information."
"""
        
        # Get answer from LLM
        answer = self._call_llm(prompt, system_prompt)
        
        # Get sources
        sources = self.retriever.get_sources(chunks)
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'has_context': True,
            'num_chunks_used': len(chunks)
        }
    
    def answer_with_context(self, question: str, context: str) -> str:
        """
        Answer a question with provided context (no retrieval).
        
        Args:
            question: User question
            context: Context text to use
            
        Returns:
            Answer string
        """
        system_prompt = """You are a helpful technical documentation assistant.
Answer questions based on the provided context. Be concise and accurate."""
        
        prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        
        return self._call_llm(prompt, system_prompt)
    
    def is_ready(self) -> bool:
        """Check if the QA engine is ready to answer questions."""
        return (self.client is not None and 
                self.retriever.vector_store.is_initialized())


if __name__ == "__main__":
    # Test the QA engine
    qa = QAEngine()
    
    if qa.is_ready():
        test_questions = [
            "How do I create a FastAPI route?",
            "What are path parameters in FastAPI?",
            "How do I run a FastAPI server?"
        ]
        
        for q in test_questions:
            print(f"\n{'='*60}")
            print(f"Q: {q}")
            result = qa.answer(q)
            print(f"A: {result['answer']}")
            if result['sources']:
                print(f"Sources: {[s['title'] for s in result['sources']]}")
    else:
        print("QA engine not ready. Check API keys and run ingestion first.")
