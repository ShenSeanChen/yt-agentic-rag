# Copyright 2024
# Directory: yt-agentic-rag/app/services/embedding.py

"""
Embedding service for generating text embeddings using OpenAI.
Focused solely on vector embeddings for RAG retrieval.
"""

import logging
from typing import List
import openai
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """Service for generating text embeddings using OpenAI."""
    
    def __init__(self):
        """Initialize OpenAI client for embeddings."""
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
        self.embed_model = settings.openai_embed_model
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (1536 dimensions for text-embedding-3-small)
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embed_model,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.embed_texts([query])
        return embeddings[0]


# Global service instance
embedding_service = EmbeddingService()
