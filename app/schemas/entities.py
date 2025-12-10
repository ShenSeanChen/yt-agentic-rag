# Copyright 2024
# Directory: yt-agentic-rag/app/schemas/entities.py

"""
Database Entity Schemas - Models Matching Supabase Tables.

Pydantic models that mirror the database schema:
- RagChunk: Represents a document chunk with embedding in rag_chunks table
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class RagChunk(BaseModel):
    """Database entity for a RAG chunk."""
    id: Optional[int] = Field(None, description="Database primary key")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    source: str = Field(..., description="Source URL or identifier")
    text: str = Field(..., description="Text content")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    
    # Additional fields for search results
    similarity: Optional[float] = Field(None, description="Similarity score (0-1)")
    distance: Optional[float] = Field(None, description="Cosine distance")


class SearchResult(BaseModel):
    """Search result with chunk and metadata."""
    chunk: RagChunk = Field(..., description="The retrieved chunk")
    similarity: float = Field(..., description="Similarity score")
    rank: int = Field(..., description="Result rank (1-based)")
