"""
Retriever Module
Retrieves relevant document chunks for a given query.
"""

import os
from typing import List, Dict, Optional
from .embeddings import EmbeddingModel
from .vector_store import VectorStore


class Retriever:
    """Retrieves relevant chunks using semantic similarity."""
    
    def __init__(self, 
                 embedding_model: EmbeddingModel = None,
                 vector_store: VectorStore = None,
                 data_dir: str = "data"):
        """
        Initialize the retriever.
        
        Args:
            embedding_model: EmbeddingModel instance (creates new if None)
            vector_store: VectorStore instance (creates new if None)
            data_dir: Directory for data storage
        """
        self.data_dir = data_dir
        self.embedding_model = embedding_model or EmbeddingModel(data_dir=data_dir)
        self.vector_store = vector_store or VectorStore(data_dir=data_dir)
    
    def retrieve(self, 
                 query: str, 
                 top_k: int = 3,
                 score_threshold: float = 0.3) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        # Check if index is ready
        if not self.vector_store.is_initialized():
            print("⚠️ Vector store not initialized. Please run ingestion first.")
            return []
        
        # Get query embedding
        query_embedding = self.embedding_model.get_query_embedding(query)
        
        # Search for similar chunks
        results = self.vector_store.search(
            query_embedding, 
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        return results
    
    def format_context(self, chunks: List[Dict], include_sources: bool = True) -> str:
        """
        Format retrieved chunks into a context string for the LLM.
        
        Args:
            chunks: List of retrieved chunks
            include_sources: Whether to include source information
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            text = chunk['text']
            
            if include_sources:
                source = chunk.get('source', 'Unknown')
                title = chunk.get('title', 'Untitled')
                context_parts.append(
                    f"[Source {i}: {title}]\n{text}\n"
                )
            else:
                context_parts.append(f"[Document {i}]\n{text}\n")
        
        return "\n".join(context_parts)
    
    def get_sources(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract source information from chunks.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            List of source dictionaries
        """
        sources = []
        seen = set()
        
        for chunk in chunks:
            source_url = chunk.get('source', '')
            if source_url and source_url not in seen:
                seen.add(source_url)
                sources.append({
                    'title': chunk.get('title', 'Untitled'),
                    'url': source_url,
                    'doc_name': chunk.get('doc_name', 'unknown')
                })
        
        return sources


if __name__ == "__main__":
    # Test the retriever
    retriever = Retriever()
    
    test_queries = [
        "How do I create a FastAPI route?",
        "What are path parameters?",
        "How to validate request data?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = retriever.retrieve(query)
        
        if results:
            print(f"   Found {len(results)} relevant chunks:")
            for r in results:
                print(f"   - [{r['score']:.3f}] {r['title']}: {r['text'][:100]}...")
        else:
            print("   No results found")
