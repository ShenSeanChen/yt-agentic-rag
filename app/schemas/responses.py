# Copyright 2024
# Directory: yt-agentic-rag/app/schemas/responses.py

"""
API Response Schemas - Output Formats for Endpoints.

Pydantic models that define API response structures:
- AnswerResponse: Traditional RAG response with citations
- AgentResponse: Agentic response with tool calls and results
- HealthResponse: System health check response
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Shared Models
# ============================================================================

class SeedResponse(BaseModel):
    """Response model for the /seed endpoint."""
    inserted: int = Field(..., description="Number of chunks successfully inserted")


class HealthResponse(BaseModel):
    """Response model for the /healthz endpoint."""
    status: str = Field(..., description="Health status (ok, degraded, error)")
    database_connected: bool = Field(..., description="Database connection status")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type/category")
    detail: str = Field(..., description="Detailed error information")


# ============================================================================
# Traditional RAG Response Models (/answer endpoint)
# ============================================================================

class DebugInfo(BaseModel):
    """Debug information for traditional RAG responses."""
    top_doc_ids: List[str] = Field(..., description="IDs of retrieved chunks")
    latency_ms: int = Field(..., description="Total processing time in milliseconds")


class AnswerResponse(BaseModel):
    """
    Response model for the /answer endpoint (traditional RAG).
    Returns answer with citations but no tool execution.
    """
    text: str = Field(..., description="Generated answer with inline citations")
    citations: List[str] = Field(..., description="List of chunk IDs used as sources")
    debug: DebugInfo = Field(..., description="Debug information")


# ============================================================================
# Agentic RAG Response Models (/agent endpoint)
# ============================================================================

class ToolCallInfo(BaseModel):
    """Information about a tool call made by the agent."""
    tool_name: str = Field(..., description="Name of the tool that was called")
    arguments: Dict[str, Any] = Field(
        ..., 
        description="Arguments passed to the tool"
    )
    call_id: str = Field(..., description="Unique identifier for this tool call")


class ToolResultInfo(BaseModel):
    """Result from a tool execution."""
    call_id: str = Field(..., description="ID of the tool call this result is for")
    tool_name: str = Field(..., description="Name of the tool that was executed")
    success: bool = Field(..., description="Whether the tool execution succeeded")
    result: Optional[Dict[str, Any]] = Field(
        None, 
        description="Tool result data if successful"
    )
    error: Optional[str] = Field(
        None, 
        description="Error message if execution failed"
    )


class AgentDebugInfo(BaseModel):
    """Debug information for agent responses."""
    rag_context_used: bool = Field(
        ..., 
        description="Whether RAG context was retrieved and available"
    )
    rag_chunk_ids: List[str] = Field(
        ..., 
        description="IDs of RAG chunks that were retrieved"
    )
    tools_called: List[str] = Field(
        ..., 
        description="Names of tools that were executed"
    )
    iterations: int = Field(
        ..., 
        description="Number of agent loop iterations"
    )
    latency_ms: int = Field(
        ..., 
        description="Total processing time in milliseconds"
    )


class AgentResponse(BaseModel):
    """
    Response model for the /agent endpoint (Agentic RAG with tools).
    
    This response includes:
    - text: The final response from the agent
    - tool_calls: List of tools that were called during processing
    - tool_results: Results from each tool execution
    - citations: RAG chunk IDs that were cited in the response
    - debug: Debug information including timing and context usage
    
    Example response:
    {
        "text": "I've scheduled a 30-minute consultation call...",
        "tool_calls": [{"tool_name": "create_calendar_event", ...}],
        "tool_results": [{"success": true, "result": {"event_id": "abc123"}}],
        "citations": ["scheduling_consultation_v1"],
        "debug": {"rag_context_used": true, "tools_called": ["create_calendar_event"], ...}
    }
    """
    text: str = Field(
        ..., 
        description="Final response text from the agent"
    )
    tool_calls: List[ToolCallInfo] = Field(
        default_factory=list,
        description="List of tool calls made during processing"
    )
    tool_results: List[ToolResultInfo] = Field(
        default_factory=list,
        description="Results from tool executions"
    )
    citations: List[str] = Field(
        default_factory=list,
        description="List of RAG chunk IDs cited in the response"
    )
    debug: AgentDebugInfo = Field(
        ..., 
        description="Debug information about the agent's processing"
    )
