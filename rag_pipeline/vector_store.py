"""
Vector Store Module
FAISS-based vector storage and similarity search.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional


class VectorStore:
    """FAISS vector store for similarity search."""
    
    def __init__(self, 
                 dimension: int = 384,
                 data_dir: str = "data"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Embedding dimension
            data_dir: Directory to save/load index
        """
        self.dimension = dimension
        self.data_dir = data_dir
        self.index = None
        self.chunks = []
        
        os.makedirs(data_dir, exist_ok=True)
    
    def build_index(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            chunks: List of chunk dictionaries (stored for retrieval)
        """
        import faiss
        
        print(f"🏗️ Building FAISS index with {len(embeddings)} vectors...")
        
        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))
        
        # Update dimension from actual embeddings
        self.dimension = embeddings.shape[1]
        
        # Create index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity after normalization)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store chunks
        self.chunks = chunks
        
        # Save to disk
        self._save()
        
        print(f"✅ Index built with {self.index.ntotal} vectors")
    
    def search(self, 
               query_embedding: np.ndarray, 
               top_k: int = 3,
               score_threshold: float = 0.3) -> List[Dict]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of chunks with similarity scores
        """
        import faiss
        
        if self.index is None:
            self._load()
        
        if self.index is None or self.index.ntotal == 0:
            print("⚠️ Index is empty")
            return []
        
        # Prepare query
        query_embedding = np.ascontiguousarray(
            query_embedding.reshape(1, -1).astype('float32')
        )
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= score_threshold:
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(score)
                results.append(chunk)
        
        return results
    
    def _save(self) -> None:
        """Save index and chunks to disk."""
        import faiss
        
        # Save FAISS index
        index_path = os.path.join(self.data_dir, 'faiss_index.bin')
        faiss.write_index(self.index, index_path)
        
        # Save chunks
        chunks_path = os.path.join(self.data_dir, 'chunks_indexed.json')
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        meta_path = os.path.join(self.data_dir, 'index_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dimension': self.dimension,
                'num_vectors': self.index.ntotal,
                'num_chunks': len(self.chunks)
            }, f, indent=2)
        
        print(f"💾 Saved index to {self.data_dir}")
    
    def _load(self) -> bool:
        """Load index and chunks from disk."""
        import faiss
        
        index_path = os.path.join(self.data_dir, 'faiss_index.bin')
        chunks_path = os.path.join(self.data_dir, 'chunks_indexed.json')
        
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            return False
        
        print(f"📂 Loading index from {self.data_dir}...")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print(f"✅ Loaded {self.index.ntotal} vectors")
        return True
    
    def is_initialized(self) -> bool:
        """Check if index is initialized."""
        if self.index is None:
            self._load()
        return self.index is not None and self.index.ntotal > 0


if __name__ == "__main__":
    # Test the vector store
    store = VectorStore()
    
    # Check if index exists
    if store.is_initialized():
        print(f"Index has {store.index.ntotal} vectors")
    else:
        print("No index found. Run the full pipeline first.")
