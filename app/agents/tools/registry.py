# Directory: yt-agentic-rag/app/agents/tools/registry.py

"""
Tool Registry - Central Hub for Agent Capabilities.

Provides a centralized way to register, discover, and execute tools.
This is where the agent looks up which tools are available and how to run them.
"""

import logging
from typing import Dict, Any, Optional, List

from .base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for all available agent tools.
    
    The registry pattern allows for:
    - Easy addition of new tools
    - Centralized tool discovery
    - Consistent tool execution interface
    
    Usage:
        # Get the global registry instance
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
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """
        Ensure default tools are registered.
        Uses lazy initialization to avoid circular imports.
        """
        if not self._initialized:
            self._register_default_tools()
            self._initialized = True
    
    def _register_default_tools(self) -> None:
        """
        Register all default tools.
        Import tools here to avoid circular imports at module load time.
        """
        # Import tools lazily to avoid circular imports
        from .calendar_tool import calendar_tool
        from .email_tool import email_tool
        
        self.register(calendar_tool)
        self.register(email_tool)
        
        logger.info(
            f"Registered {len(self._tools)} default tools: {list(self._tools.keys())}"
        )
    
    def register(self, tool: BaseTool) -> None:
        """
        Register a new tool with the registry.
        
        Args:
            tool: Tool instance to register (must inherit from BaseTool)
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            logger.warning(
                f"Tool '{tool.name}' already registered, overwriting"
            )
        
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.
        
        Args:
            name: Name of the tool to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool instance by name.
        
        Args:
            name: Tool name to look up
            
        Returns:
            Tool instance or None if not found
        """
        self._ensure_initialized()
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """
        Get list of all registered tool names.
        
        Returns:
            List of tool name strings
        """
        self._ensure_initialized()
        return list(self._tools.keys())
    
    def get_tool_info(self) -> List[Dict[str, str]]:
        """
        Get information about all registered tools.
        
        Returns:
            List of dicts with tool name and description
        """
        self._ensure_initialized()
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]
    
    async def execute_tool(
        self, 
        tool_name: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool by name with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Tool execution result dict with:
            - success: bool
            - result: dict if successful
            - error: str if failed
        """
        self._ensure_initialized()
        
        tool = self.get_tool(tool_name)
        
        if not tool:
            logger.error(f"Tool not found: {tool_name}")
            return {
                "success": False,
                "result": None,
                "error": f"Unknown tool: {tool_name}. Available tools: {self.list_tools()}"
            }
        
        try:
            logger.info(
                f"Executing tool: {tool_name} with args: "
                f"{list(kwargs.keys())}"
            )
            result = await tool.execute(**kwargs)
            logger.info(
                f"Tool '{tool_name}' execution completed: "
                f"success={result.get('success')}"
            )
            return result
            
        except Exception as e:
            logger.error(
                f"Tool execution failed: {tool_name} - {str(e)}", 
                exc_info=True
            )
            return {
                "success": False,
                "result": None,
                "error": f"Tool execution failed: {str(e)}"
            }
    
    def is_valid_tool(self, name: str) -> bool:
        """
        Check if a tool name is valid/registered.
        
        Args:
            name: Tool name to check
            
        Returns:
            True if tool is registered
        """
        self._ensure_initialized()
        return name in self._tools


# Global registry instance - use this throughout the application
tool_registry = ToolRegistry()

