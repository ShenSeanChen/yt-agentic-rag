# Directory: yt-agentic-rag/app/agents/tools/base.py

"""
Base Tool Interface - Template for Creating New Agent Tools.

All tools should inherit from BaseTool to ensure:
- Consistent interface for the agent orchestrator
- Standardized success/error response formats
- Parameter validation helpers

To create a new tool, inherit from BaseTool and implement:
- name: str property
- description: str property  
- execute(**kwargs): async method that performs the action
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.
    
    To create a new tool:
    1. Create a new file in app/services/tools/
    2. Inherit from BaseTool
    3. Implement name, description, and execute() method
    4. Register the tool in the registry
    
    Example:
        class MyTool(BaseTool):
            @property
            def name(self) -> str:
                return "my_tool"
            
            @property
            def description(self) -> str:
                return "Does something useful"
            
            async def execute(self, **kwargs) -> Dict[str, Any]:
                # Implementation here
                return {"success": True, "result": {...}}
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name identifier for the tool.
        Must match the name in TOOL_DEFINITIONS.
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Dict with structure:
            {
                "success": bool,
                "result": {...} if success else None,
                "error": str if not success else None
            }
        """
        pass
    
    def validate_params(
        self, 
        required: List[str], 
        provided: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Validate that all required parameters are provided.
        
        Args:
            required: List of required parameter names
            provided: Dict of provided parameters
            
        Returns:
            Tuple of (is_valid, missing_params)
        """
        missing = [
            param for param in required 
            if param not in provided or provided[param] is None
        ]
        
        if missing:
            logger.warning(
                f"Tool '{self.name}' missing required params: {missing}"
            )
            return False, missing
        
        return True, []
    
    def _success_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a standardized success response.
        
        Args:
            result: The result data to return
            
        Returns:
            Standardized success response dict
        """
        return {
            "success": True,
            "result": result,
            "error": None
        }
    
    def _error_response(self, error: str) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            error: Error message
            
        Returns:
            Standardized error response dict
        """
        logger.error(f"Tool '{self.name}' error: {error}")
        return {
            "success": False,
            "result": None,
            "error": error
        }

