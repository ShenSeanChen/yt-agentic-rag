# Copyright 2024
# Directory: yt-agentic-rag/app/main.py

"""
FastAPI application for Agentic RAG (Retrieval-Augmented Generation) backend.

This application provides:
- Traditional RAG endpoints for question answering (/answer)
- Agentic RAG endpoints with tool calling (/agent)
- Document seeding for the knowledge base (/seed)
- Health checks and API documentation

The key difference between /answer and /agent:
- /answer: Retrieve context → Generate answer (RAG only)
- /agent: Retrieve context → Reason → Execute tools → Generate answer (Agentic RAG)
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse

from .config.settings import get_settings
from .config.database import db
from .schemas.requests import SeedRequest, AnswerRequest, AgentRequest
from .schemas.responses import (
    SeedResponse, 
    AnswerResponse, 
    AgentResponse,
    HealthResponse, 
    AgentDebugInfo,
    ToolCallInfo, 
    ToolResultInfo
)
from .services.rag import rag_service
from .agents.orchestrator import agent_service
from .data.default_documents import DEFAULT_DOCUMENTS
from .schemas.tool_schemas import TOOL_DEFINITIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events for startup and shutdown."""
    # Startup
    logger.info("Starting Agentic RAG API application")
    
    try:
        # Initialize database connection
        await db.connect()
        
        # Check database schema
        await db.initialize_schema()
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agentic RAG API application")
    await db.disconnect()


# Create FastAPI application
app = FastAPI(
    title="Agentic RAG AI Backend",
    description="""
A production-ready FastAPI backend demonstrating **Agentic RAG** - combining 
Retrieval-Augmented Generation with autonomous tool-calling capabilities.

## Features
- **RAG Pipeline**: Vector similarity search with Supabase pgvector
- **Agent Tools**: Google Calendar scheduling, Email sending
- **Multi-AI Provider**: OpenAI & Anthropic support
- **Citation-based answers**: Source tracking for all responses

## Key Endpoints
- `POST /answer` - Traditional RAG (retrieve + answer)
- `POST /agent` - Agentic RAG (retrieve + reason + act + answer)
- `GET /tools` - List available agent tools

## What's the difference?
- `/answer`: Good for Q&A - retrieves context and generates answers
- `/agent`: Good for actions - can schedule meetings, send emails, AND answer questions
    """,
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # NextJS default
        "http://localhost:3001",  # Alternative port
        "https://yt-agentic-rag.vercel.app",  # Production frontend
        "*"  # Allow all for development (restrict in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the chat interface
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================================
# General Endpoints
# ============================================================================

@app.get("/chat", tags=["General"])
async def chat_interface():
    """Serve the chat interface HTML page."""
    return FileResponse("static/chat.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Handle favicon requests to prevent 404 errors."""
    return Response(status_code=204)  # No Content


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information and available endpoints."""
    return {
        "message": "Welcome to Agentic RAG AI Backend",
        "version": "2.0.0",
        "description": "RAG + Tool Calling = Agentic RAG",
        "docs": "/docs",
        "endpoints": {
            "health": "/healthz",
            "seed": "/seed (POST) - Seed knowledge base",
            "answer": "/answer (POST) - Traditional RAG Q&A",
            "agent": "/agent (POST) - Agentic RAG with tools",
            "documents": "/documents (GET) - View default documents",
            "tools": "/tools (GET) - List available tools"
        }
    }


@app.get("/greet/{name}", tags=["General"])
async def greet(name: str):
    """
    Greet endpoint that returns a personalized greeting.
    
    Args:
        name: Name of the person to greet
        
    Returns:
        Personalized greeting message
    """
    return {"message": f"Hello, {name}! Welcome to Agentic RAG!"}


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint with database connectivity test.
    
    Returns:
        Health status and database connection state
    """
    try:
        db_healthy = await db.health_check()
        
        return HealthResponse(
            status="ok" if db_healthy else "degraded",
            database_connected=db_healthy
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {str(e)}"
        )


# ============================================================================
# Knowledge Base Endpoints
# ============================================================================

@app.get("/documents", tags=["Knowledge Base"])
async def get_documents():
    """
    Get the default documents used for seeding the knowledge base.
    
    Returns:
        List of default documents with chunk_id, source, and text
    """
    return {"documents": DEFAULT_DOCUMENTS}


@app.post("/seed", response_model=SeedResponse, tags=["Knowledge Base"])
async def seed_documents(request: SeedRequest = SeedRequest()):
    """
    Seed the knowledge base with documents.
    
    If no documents are provided, seeds with default policy/FAQ/scheduling documents.
    Documents are chunked, embedded, and stored in the vector database.
    
    Args:
        request: Optional list of documents to seed
        
    Returns:
        Number of chunks successfully inserted
    """
    try:
        logger.info("Starting document seeding process")
        
        # Convert request documents to the format expected by RAG service
        documents = None
        if request.docs:
            documents = [
                {
                    'chunk_id': doc.chunk_id,
                    'source': doc.source,
                    'text': doc.text
                }
                for doc in request.docs
            ]
        
        # Process documents through RAG pipeline
        inserted_count = await rag_service.seed_documents(documents)
        
        logger.info(f"Seeding completed: {inserted_count} chunks inserted")
        
        return SeedResponse(inserted=inserted_count)
        
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to seed documents: {str(e)}"
        )


# ============================================================================
# RAG Endpoints (Traditional)
# ============================================================================

@app.post("/answer", response_model=AnswerResponse, tags=["RAG"])
async def answer_question(request: AnswerRequest):
    """
    Answer a question using traditional RAG (no tool calling).
    
    Pipeline:
    1. Embed the query
    2. Vector similarity search to find relevant chunks
    3. Generate answer using LLM with context
    4. Return answer with citations
    
    Use this endpoint for pure Q&A without actions.
    For actions (scheduling, email), use /agent instead.
    
    Args:
        request: Query and optional top_k parameter
        
    Returns:
        Generated answer with citations and debug info
    """
    try:
        logger.info(f"Processing RAG query: '{request.query[:100]}...'")
        
        result = await rag_service.answer_query(
            query=request.query,
            top_k=request.top_k
        )
        
        response = AnswerResponse(
            text=result['text'],
            citations=result['citations'],
            debug={
                'top_doc_ids': result['debug']['top_doc_ids'],
                'latency_ms': result['debug']['latency_ms']
            }
        )
        
        logger.info(f"RAG query processed in {result['debug']['latency_ms']}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"RAG query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


# ============================================================================
# Agent Endpoints (Agentic RAG with Tools)
# ============================================================================

@app.get("/tools", tags=["Agent"])
async def list_tools():
    """
    List all available agent tools.
    
    Returns:
        List of tools with name and description
    """
    return {
        "tools": [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"]
            }
            for t in TOOL_DEFINITIONS
        ]
    }


@app.post("/agent", response_model=AgentResponse, tags=["Agent"])
async def agent_query(request: AgentRequest):
    """
    Process a query using Agentic RAG (with tool calling).
    
    Pipeline:
    1. Retrieve relevant context from RAG
    2. Agent decides if tools are needed based on query + context
    3. Execute tools if necessary (calendar, email, etc.)
    4. Generate final response with citations and tool results
    
    This endpoint can:
    - Answer questions (like /answer)
    - Take actions (schedule meetings, send emails)
    - Use RAG context to inform tool parameters
    
    Example queries:
    - "What is your return policy?" → RAG answer
    - "Schedule a meeting with john@example.com for Tuesday at 2pm" → Tool call
    - "Schedule a standard consultation with John" → RAG (get duration) + Tool call
    
    Args:
        request: Query, optional user_id, and top_k
        
    Returns:
        Response with text, tool_calls, tool_results, citations, and debug info
    """
    try:
        logger.info(f"Processing agent query: '{request.query[:100]}...'")
        
        # Convert chat_history from Pydantic models to dicts if provided
        chat_history = None
        if request.chat_history:
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.chat_history
            ]
        
        result = await agent_service.process_query(
            query=request.query,
            chat_history=chat_history,
            user_id=request.user_id,
            top_k=request.top_k
        )
        
        # Build response with proper model instances
        response = AgentResponse(
            text=result['text'],
            tool_calls=[
                ToolCallInfo(**tc) for tc in result.get('tool_calls', [])
            ],
            tool_results=[
                ToolResultInfo(**tr) for tr in result.get('tool_results', [])
            ],
            citations=result.get('citations', []),
            debug=AgentDebugInfo(**result['debug'])
        )
        
        logger.info(
            f"Agent query processed in {result['debug']['latency_ms']}ms, "
            f"tools called: {result['debug'].get('tools_called', [])}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Agent query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process agent query: {str(e)}"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors with helpful information."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": "The requested endpoint does not exist",
            "available_endpoints": [
                "/", "/healthz", "/seed", "/answer", "/agent", "/tools", "/docs"
            ]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please check the logs."
        }
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
