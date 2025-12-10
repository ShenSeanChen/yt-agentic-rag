# Directory: yt-agentic-rag/app/agents/tools/__init__.py

"""
Agent Tools - Capabilities the AI Agent Can Execute.

This module provides the tools that make the agent "agentic":
- tool_registry: Central registry for discovering and executing tools
- BaseTool: Abstract base class for creating new tools
- Individual tool instances (calendar_tool, email_tool)

Usage:
    from app.agents.tools import tool_registry
    
    # Execute a tool
    result = await tool_registry.execute_tool(
        "create_calendar_event",
        summary="Meeting",
        start_datetime="2024-12-15T14:00:00",
        end_datetime="2024-12-15T15:00:00"
    )
    
    # List available tools
    tools = tool_registry.list_tools()

Adding new tools:
    1. Create a new file in this directory (e.g., my_tool.py)
    2. Create a class inheriting from BaseTool
    3. Implement name, description, and execute() method
    4. Create a global instance
    5. Import and register in registry.py
    6. Add tool definition to app/schemas/tool_schemas.py
"""

from .base import BaseTool
from .registry import tool_registry
from .calendar_tool import calendar_tool
from .email_tool import email_tool

__all__ = [
    'BaseTool',
    'tool_registry',
    'calendar_tool',
    'email_tool',
]

