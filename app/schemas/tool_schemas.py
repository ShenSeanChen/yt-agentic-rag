# Directory: yt-agentic-rag/app/schemas/tool_schemas.py

"""
Tool Schemas - LLM Function Calling Definitions.

This file defines the tools available to the agent in OpenAI's
function-calling format. The LLM uses these definitions to:
- Understand what tools are available
- Know what parameters each tool requires
- Generate valid tool calls

When adding a new tool, add its definition to TOOL_DEFINITIONS.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class ToolName(str, Enum):
    """
    Available tool names in the system.
    Add new tools here as they are implemented.
    """
    CREATE_CALENDAR_EVENT = "create_calendar_event"
    SEND_EMAIL = "send_email"
    # Future tools can be added here:
    # SEARCH_AVAILABILITY = "search_availability"
    # UPDATE_USER_PROFILE = "update_user_profile"
    # LOOKUP_CUSTOMER_INFO = "lookup_customer_info"


class ToolParameter(BaseModel):
    """Schema for a single tool parameter definition."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, number, array, etc.)")
    description: str = Field(..., description="Description of the parameter")
    required: bool = Field(default=True, description="Whether the parameter is required")
    enum: Optional[List[str]] = Field(default=None, description="Allowed values if restricted")


class ToolDefinition(BaseModel):
    """Schema for defining a tool available to the agent."""
    name: ToolName = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Human-readable description of the tool")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")


class ToolCall(BaseModel):
    """Represents a tool call made by the agent."""
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    call_id: str = Field(
        default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this tool call"
    )


class ToolResult(BaseModel):
    """Result from executing a tool."""
    call_id: str = Field(..., description="ID of the tool call this result is for")
    tool_name: str = Field(..., description="Name of the tool that was executed")
    success: bool = Field(..., description="Whether the tool execution succeeded")
    result: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Tool result data if successful"
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error message if execution failed"
    )


# ============================================================================
# TOOL DEFINITIONS - OpenAI Function Calling Format
# ============================================================================
# These definitions tell the LLM what tools are available and how to use them.
# Format follows OpenAI's function calling specification.

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": (
                "Create a calendar event/meeting on Google Calendar. "
                "Use this when the user wants to schedule a meeting, appointment, or call. "
                "If the user mentions a 'standard consultation' or similar, check the RAG context "
                "for default durations before setting the end time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Title/summary of the event (e.g., 'Consultation Call with John')"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the event (optional)"
                    },
                    "start_datetime": {
                        "type": "string",
                        "description": (
                            "Start date and time in ISO 8601 format "
                            "(e.g., '2024-12-15T14:00:00')"
                        )
                    },
                    "end_datetime": {
                        "type": "string",
                        "description": (
                            "End date and time in ISO 8601 format "
                            "(e.g., '2024-12-15T15:00:00')"
                        )
                    },
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of attendee email addresses"
                    },
                    "timezone": {
                        "type": "string",
                        "description": (
                            "Timezone for the event (e.g., 'America/New_York'). "
                            "Defaults to UTC if not specified."
                        )
                    }
                },
                "required": ["summary", "start_datetime", "end_datetime"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Send an email to a recipient. Use this when the user wants to send "
                "a confirmation, follow-up, notification, or any email communication."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content (plain text)"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get all available tool definitions for the LLM.
    
    Returns:
        List of tool definitions in OpenAI function-calling format.
    """
    return TOOL_DEFINITIONS


def get_tool_names() -> List[str]:
    """
    Get list of all available tool names.
    
    Returns:
        List of tool name strings.
    """
    return [t["function"]["name"] for t in TOOL_DEFINITIONS]

