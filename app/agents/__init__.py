# Directory: yt-agentic-rag/app/agents/__init__.py

"""
Agents Module - The Heart of Agentic RAG.

This module contains:
- orchestrator.py: The main agent reasoning loop
- tools/: Agent capabilities (calendar, email, etc.)

The agent orchestrator is what makes this "Agentic RAG" - it can:
1. Retrieve context from RAG
2. Reason about whether tools are needed
3. Execute tools to take real-world actions
4. Generate responses with citations
"""

from .orchestrator import agent_service

__all__ = ['agent_service']

