# Copyright 2024
# Directory: yt-agentic-rag/app/schemas/requests.py

"""
API Request Schemas - Input Validation for Endpoints.

Pydantic models that define and validate incoming API requests:
- SeedRequest: For seeding the knowledge base
- AnswerRequest: For traditional RAG queries
- AgentRequest: For agentic queries with tool calling
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in the chat history."""
    role: Literal["user", "assistant"] = Field(
        ..., 
        description="Role of the message sender"
    )
    content: str = Field(..., description="Message content")


class DocumentChunk(BaseModel):
    """Individual document chunk for seeding the knowledge base."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    source: str = Field(..., description="Source URL or identifier")
    text: str = Field(..., description="Text content of the chunk")


class SeedRequest(BaseModel):
    """Request model for the /seed endpoint."""
    docs: Optional[List[DocumentChunk]] = Field(
        None, 
        description="Optional list of documents to seed. If omitted, uses default documents."
    )


class AnswerRequest(BaseModel):
    """
    Request model for the /answer endpoint (traditional RAG).
    Use this for question-answering without tool execution.
    """
    query: str = Field(..., description="User question to answer")
    top_k: Optional[int] = Field(
        6, 
        ge=1, 
        le=20, 
        description="Number of chunks to retrieve for context"
    )


class AgentRequest(BaseModel):
    """
    Request model for the /agent endpoint (Agentic RAG with tools).
    
    Use this when you want the agent to:
    - Answer questions using RAG context
    - Execute actions (schedule meetings, send emails)
    - Combine both capabilities in a single request
    
    Example requests:
    - "What is your return policy?" (RAG only)
    - "Schedule a meeting with john@example.com for Tuesday at 2pm" (Tool call)
    - "Schedule a standard consultation call with John" (RAG + Tool: uses RAG to determine duration)
    
    For multi-turn conversations, include chat_history to maintain context.
    """
    query: str = Field(
        ..., 
        description="User question or action request"
    )
    chat_history: Optional[List[ChatMessage]] = Field(
        None,
        description="Optional chat history for multi-turn conversations"
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier for personalization and tracking"
    )
    top_k: Optional[int] = Field(
        6,
        ge=1,
        le=20,
        description="Number of RAG chunks to retrieve for context"
    )
