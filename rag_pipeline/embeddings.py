"""
Embedding Model Module
Creates vector embeddings for text chunks using SentenceTransformers.
"""

import os
import json
import numpy as np
from typing import List, Union, Optional
from tqdm import tqdm


class EmbeddingModel:
    """Creates embeddings using SentenceTransformers."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 data_dir: str = "data"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model
            data_dir: Directory to save embeddings
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        os.makedirs(data_dir, exist_ok=True)
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            print(f"📦 Loading embedding model: {self.model_name}...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✅ Model loaded (dimension: {self.embedding_dim})")
    
    def encode(self, texts: Union[str, List[str]], show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        print(f"🔢 Creating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_chunks(self, chunks: List[dict]) -> np.ndarray:
        """
        Create embeddings for a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.encode(texts)
        
        # Save embeddings
        self._save_embeddings(embeddings)
        
        print(f"✅ Created {len(embeddings)} embeddings")
        return embeddings
    
    def _save_embeddings(self, embeddings: np.ndarray) -> str:
        """Save embeddings to disk."""
        filepath = os.path.join(self.data_dir, 'embeddings.npy')
        np.save(filepath, embeddings)
        return filepath
    
    def load_embeddings(self) -> Optional[np.ndarray]:
        """Load previously saved embeddings."""
        filepath = os.path.join(self.data_dir, 'embeddings.npy')
        
        if not os.path.exists(filepath):
            return None
        
        return np.load(filepath)
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        self._load_model()
        return self.model.encode([query], convert_to_numpy=True)[0]


if __name__ == "__main__":
    # Test the embedding model
    model = EmbeddingModel()
    
    # Test encoding
    test_texts = [
        "FastAPI is a modern Python web framework",
        "You can create API endpoints with decorators",
        "Path parameters allow dynamic URLs"
    ]
    
    embeddings = model.encode(test_texts)
    print(f"\nCreated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
