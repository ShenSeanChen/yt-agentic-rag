# Copyright 2024
# Directory: yt-agentic-rag/app/services/chunker.py

"""
Text chunking utilities for RAG document processing.
Implements simple word-based chunking with overlap for optimal retrieval.
"""

import logging
import re
from typing import List, Dict, Any
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TextChunker:
    """Simple text chunker with word-based splitting and overlap."""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        """
        Initialize chunker with configuration.
        
        Args:
            chunk_size: Approximate tokens per chunk (default from settings)
            overlap: Overlap tokens between chunks (default from settings)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
        
        # Rough approximation: 1 token ≈ 0.75 words
        self.words_per_chunk = int(self.chunk_size * 0.75)
        self.overlap_words = int(self.overlap * 0.75)
        
        logger.info(f"Initialized chunker: ~{self.words_per_chunk} words per chunk, {self.overlap_words} overlap")
    
    def chunk_text(self, text: str, source: str, base_chunk_id: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            source: Source identifier (URL, filename, etc.)
            base_chunk_id: Base identifier for chunks (will append #1, #2, etc.)
            
        Returns:
            List of chunk dictionaries with chunk_id, source, and text
        """
        if not text or not text.strip():
            return []
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        words = cleaned_text.split()
        
        if len(words) <= self.words_per_chunk:
            # Text is small enough to be a single chunk
            return [{
                'chunk_id': f"{base_chunk_id}#1",
                'source': source,
                'text': cleaned_text
            }]
        
        chunks = []
        start_idx = 0
        chunk_num = 1
        
        while start_idx < len(words):
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.words_per_chunk, len(words))
            
            # Extract chunk words
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Create chunk dictionary
            chunk = {
                'chunk_id': f"{base_chunk_id}#{chunk_num}",
                'source': source,
                'text': chunk_text
            }
            chunks.append(chunk)
            
            # Move start index for next chunk (with overlap)
            if end_idx >= len(words):
                break  # We've processed all words
            
            start_idx = end_idx - self.overlap_words
            chunk_num += 1
        
        logger.info(f"Chunked text into {len(chunks)} chunks (source: {source})")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for better chunking.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents with 'text', 'source', and 'chunk_id' keys
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            text = doc.get('text', '')
            source = doc.get('source', 'unknown')
            base_chunk_id = doc.get('chunk_id', 'doc')
            
            chunks = self.chunk_text(text, source, base_chunk_id)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks




# Global chunker instance
chunker = TextChunker()
